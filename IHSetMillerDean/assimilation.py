import numpy as np
from typing import Any
from IHSetUtils.CoastlineModel import CoastlineModel
from IHSetUtils import wMOORE, wast
from .millerDean import millerDean

class assimilate_MillerDean(CoastlineModel):
    """
    Miller & Dean (2004) — EnKF parameter assimilation.
    Parameters (transformed space):
      par = [log(kero), log(kacr), Y0]
    Yini is fixed to the first observation of the split series.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Miller and Dean (2004)',
            mode='assimilation',
            model_type='CS',
            model_key='MillerDean'
        )
        self.setup_forcing()

    # ----------------------
    # Forcing & pre-processing
    # ----------------------
    def setup_forcing(self):
        cfg = self.cfg
        self.D50   = float(cfg['D50'])
        self.hberm = float(cfg['Hberm'])
        self.flagP = int(cfg['flagP'])

        # Compose sea level (surge + tide)
        self.sl   = self.surge + self.tide

        # Base sets hb, depthb in _break_waves_snell(); ensure floors
        self.tp[self.tp < 5.0]      = 5.0
        self.hb[self.hb < 0.1]      = 0.1
        self.depthb[self.depthb < .2] = 0.2

        # Mobility and transport proxies
        self.ws     = wMOORE(self.D50)
        self.Omega  = self.hb / (self.ws * self.tp)
        self.wast   = wast(self.hb, self.D50)

        # ---- Build segment counterparts for assimilation ----
        jj = self.idx_calibration  # same indices used to build *_s in calibration mode
        self.hb_s     = self.hb[jj]
        self.depthb_s = self.depthb[jj]
        self.tp_s     = self.tp[jj]
        self.sl_s     = self.sl[jj]
        self.Omega_s  = self.Omega[jj]
        self.wast_s   = self.wast[jj]

        # Initial shoreline from the first available observation (after split)
        self.Yini = float(self.Obs_splited[0])

    # ----------------------
    # Ensemble init in transformed space
    # ----------------------
    def init_par(self, population_size: int):
        # Bounds expected: lb=[kero_min, kacr_min, Y0_min], ub=[..., ..., Y0_max]
        # Sample in log-space for kero,kacr; Y0 linear.
        lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1]), self.lb[2]])
        uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1]), self.ub[2]])

        Ddim = len(lowers)
        pop = np.zeros((population_size, Ddim))
        for i in range(Ddim):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], size=population_size)
        return pop, lowers, uppers

    # ----------------------
    # One EnKF step: forecast last value over the obs segment
    # ----------------------
    def model_step(self, par: np.ndarray, t_idx: int, context: Any | None = None):
        kero = float(np.exp(par[0]))
        kacr = float(np.exp(par[1]))
        Y0   = float(par[2])

        # segment indices for this obs step
        i0, i1   = self.idx_obs_splited[t_idx - 1], self.idx_obs_splited[t_idx]
        hb_seg     = self.hb_s[i0:i1]
        depthb_seg = self.depthb_s[i0:i1]
        sl_seg     = self.sl_s[i0:i1]
        wast_seg   = self.wast_s[i0:i1]
        dt_seg     = self.dt_s[i0:i1]
        Omega_seg  = self.Omega_s[i0:i1]

        # initial condition for this segment
        y0 = float(self.Yini) if (context is None or ('y_old' not in context)) else float(context['y_old'])

        Ymd, _ = millerDean(hb_seg, depthb_seg, sl_seg, wast_seg, dt_seg,
                            self.hberm, Y0, kero, kacr, y0, self.flagP, Omega_seg)
        y_last = float(Ymd[-1])
        context = {'y_old': y_last}
        return y_last, context

    # ----------------------
    # Vectorized batch step (fast path)
    # ----------------------
    def model_step_batch(self, pop: np.ndarray, t_idx: int, contexts: list[dict] | None):
        N = pop.shape[0]
        y_out   = np.empty((N,), dtype=float)
        new_ctx = [None] * N

        i0, i1   = self.idx_obs_splited[t_idx - 1], self.idx_obs_splited[t_idx]
        hb_seg     = self.hb_s[i0:i1]
        depthb_seg = self.depthb_s[i0:i1]
        sl_seg     = self.sl_s[i0:i1]
        wast_seg   = self.wast_s[i0:i1]
        dt_seg     = self.dt_s[i0:i1]
        Omega_seg  = self.Omega_s[i0:i1]

        for j in range(N):
            kero = float(np.exp(pop[j, 0]))
            kacr = float(np.exp(pop[j, 1]))
            Y0   = float(pop[j, 2])

            y0 = float(self.Yini) if (contexts is None or contexts[j] is None
                                      or ('y_old' not in contexts[j])) else float(contexts[j]['y_old'])

            Ymd, _ = millerDean(hb_seg, depthb_seg, sl_seg, wast_seg, dt_seg,
                                self.hberm, Y0, kero, kacr, y0, self.flagP, Omega_seg)
            y_last = float(Ymd[-1])
            y_out[j]   = y_last
            new_ctx[j] = {'y_old': y_last}

        return y_out, new_ctx

    # ----------------------
    # Full forward run with final parameters (for plotting/output)
    # ----------------------
    def run_model(self, par: np.ndarray) -> np.ndarray:
        # Here par is in PHYSICAL space (after _set_parameter_names)
        kero = float(par[0])
        kacr = float(par[1])
        Y0   = float(par[2])

        Ymd, _ = millerDean(self.hb, self.depthb, self.sl, self.wast, self.dt,
                            self.hberm, Y0, kero, kacr, self.Yini, self.flagP, self.Omega)
        return Ymd

    # ----------------------
    # Names & convert to physical for reporting
    # ----------------------
    def _set_parameter_names(self):
        self.par_names = ['k-', 'k+', 'Y0']
        # par_values currently in transformed space -> convert:
        kero = float(np.exp(self.par_values[0]))
        kacr = float(np.exp(self.par_values[1]))
        Y0   = float(self.par_values[2])
        self.par_values = np.array([kero, kacr, Y0], dtype=float)