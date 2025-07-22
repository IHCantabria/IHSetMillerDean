import numpy as np
from .millerDean import millerDean
from IHSetUtils import wMOORE, wast
from IHSetUtils.CoastlineModel import CoastlineModel


class cal_MillerDean_2(CoastlineModel):
    """
    cal_MillerDean_2
    
    Configuration to calibrate and run the Miller and Dean (2004) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Miller and Dean (2004)',
            mode='calibration',
            model_type='CS',
            model_key='MillerDean'
        )

        self.setup_forcing()

    def setup_forcing(self):

        self.switch_Yini = self.cfg['switch_Yini']
        self.D50 = self.cfg['D50']
        self.hberm = self.cfg['Hberm']
        self.flagP = self.cfg['flagP']
        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

        
        self.sl = self.surge + self.tide
        self.sl_s = self.surge_s + self.tide_s

        self.wast = wast(self.hb, self.D50)
        self.wast_s = wast(self.hb_s, self.D50)

        self.ws = wMOORE(self.D50)
        self.Omega = self.hb / (self.ws * self.tp)
        self.Omega_s = self.hb_s / (self.ws * self.tp_s)


    def init_par(self, population_size: int):
        if self.switch_Yini == 0:
            lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1]), self.lb[2]])
            uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1]), self.ub[2]])
        else:
            lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1]),
                               self.lb[2], 0.75*np.min(self.Obs_splited)
                               ])
            uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1]), 
                               self.ub[2], 1.25*np.min(self.Obs_splited)
                               ])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers
    
    def model_sim(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 0:
            kero = np.exp(par[0]); kacr = np.exp(par[1])
            Y0 = par[2]
            Ymd, _ = millerDean(self.hb_s, self.depthb_s,
                                self.sl_s, self.wast_s,
                                self.dt_s, self.hberm,
                                Y0, kero, kacr, self.Yini, 
                                self.flagP, self.Omega_s)
        else:
            kero = np.exp(par[0]); kacr = np.exp(par[1])
            Y0 = par[2];           Yini = par[3]
            Ymd, _ = millerDean(self.hb_s, self.depthb_s,
                                self.sl_s, self.wast_s,
                                self.dt_s, self.hberm,
                                Y0, kero, kacr, Yini, self.flagP,
                                self.Omega_s)
        return Ymd[self.idx_obs_splited]
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 0:
            kero = par[0]; kacr = par[1]
            Y0 = par[2]
            Ymd, _ = millerDean(self.hb, self.depthb,
                                self.sl, self.wast,
                                self.dt, self.hberm,
                                Y0, kero, kacr, self.Yini,
                                self.flagP, self.Omega)
        else:
            kero = par[0]; kacr = par[1]
            Y0 = par[2]; Yini = par[3]
            Ymd, _ = millerDean(self.hb, self.depthb,
                                self.sl, self.wast,
                                self.dt, self.hberm,
                                Y0, kero, kacr, Yini,
                                self.flagP, self.Omega)
        return Ymd
    
    def _set_parameter_names(self):
        if self.switch_Yini == 0:
            self.par_names = ['k-', 'k+', 'Y0']
        else:
            self.par_names = ['k-', 'k+', 'Y0', 'Yini']
        
        for idx in [0, 1]:
            self.par_values[idx] = np.exp(self.par_values[idx])

        