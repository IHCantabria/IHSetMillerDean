import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .millerDean import millerDean
import json
from IHSetUtils import wMOORE, wast, BreakingPropagation
from scipy.stats import circmean

class cal_MillerDean_3(object):
    """
    cal_MillerDean_2
    
    Configuration to calibrate and run the Miller and Dean (2004) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        self.name = 'Miller and Dean (2004)'
        self.mode = 'calibration'
        self.type = 'CS'
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['MillerDean'])
        self.cfg = cfg

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.switch_Yini = cfg['switch_Yini']
        self.lb = cfg['lb']
        self.ub = cfg['ub']
        self.D50 = cfg['D50']
        self.hberm = cfg['Hberm']
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.breakType = cfg['break_type']

        self.flagP = cfg['flagP']

        self.calibr_cfg = fo.config_cal(cfg)

        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.tp = np.mean(data.tp.values, axis=1)
            self.dir = circmean(data.dir.values, axis=1, high=360, low=0)
            self.tide = np.mean(data.tide.values, axis=1)
            self.surge = np.mean(data.surge.values, axis=1)
            self.sl = self.surge + self.tide
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
            self.depth = np.mean(data.waves_depth.values)
            self.bathy_angle = circmean(data.phi.values, high=360, low=0)
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.tp = data.tp.values[:, cfg['trs']]
            self.dir = data.dir.values[:, cfg['trs']]
            self.tide = data.tide.values[:, cfg['trs']]
            self.surge = data.surge.values[:, cfg['trs']]
            self.sl = self.surge + self.tide
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.depth = data.waves_depth.values[cfg['trs']]
            self.bathy_angle = data.phi.values[cfg['trs']]
            
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])

        data.close()
        
        if self.switch_brk == 0:
            self.depthb = self.hs/0.55
            self.hb = self.hs
            self.dirb = self.dir
        elif self.switch_brk == 1:
            self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)
        
        self.hb[self.hb < 0.1] = 0.1
        self.depthb[self.depthb < 0.2] = 0.2
        self.tp[self.tp < 5] = 5

        self.wast = wast(self.hb, self.D50)
        self.ws = wMOORE(self.D50)
        self.Omega = self.hb / (self.ws * self.tp)

        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

        

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))

        if self.switch_Yini == 0:
            def model_simulation(par):
                kero = np.exp(par[0])
                kacr = np.exp(par[1])
                Y0 = par[2]
                
                Ymd, _ = millerDean(self.hb_splited,
                                    self.depthb_splited,
                                    self.sl_splited,
                                    self.wast_splited,
                                    self.dt_splited,
                                    self.hberm,
                                    Y0,
                                    kero,
                                    kacr,
                                    self.Yini,
                                    self.flagP,
                                    self.Omega_splited)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                kero = np.exp(par[0])
                kacr = np.exp(par[1])
                Y0 = par[2]
                
                Ymd, _ = millerDean(self.hb,
                                    self.depthb,
                                    self.sl,
                                    self.wast,
                                    self.dt,
                                    self.hberm,
                                    Y0,
                                    kero,
                                    kacr,
                                    self.Yini,
                                    self.flagP,
                                    self.Omega)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1]), self.lb[2]])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1]), self.ub[2]])
                population = np.zeros((population_size, 3))
                for i in range(3):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            def model_simulation(par):
                kero = np.exp(par[0])
                kacr = np.exp(par[1])
                Y0 = par[2]
                Yini = par[3]
                
                Ymd, _ = millerDean(self.hb_splited,
                                    self.depthb_splited,
                                    self.sl_splited,
                                    self.wast_splited,
                                    self.dt_splited,
                                    self.hberm,
                                    Y0,
                                    kero,
                                    kacr,
                                    Yini,
                                    self.flagP,
                                    self.Omega_splited)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                kero = np.exp(par[0])
                kacr = np.exp(par[1])
                Y0 = par[2]
                Yini = par[3]
                
                Ymd, _ = millerDean(self.hb,
                                    self.depthb,
                                    self.sl,
                                    self.wast,
                                    self.dt,
                                    self.hberm,
                                    Y0,
                                    kero,
                                    kacr,
                                    Yini,
                                    self.flagP,
                                    self.Omega)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1]), self.lb[2], 0.75*np.min(self.Obs_splited)])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1]), self.ub[2], 1.25*np.min(self.Obs_splited)])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """ 
        ii = np.where(self.time>=self.start_date)[0][0]
        self.hb = self.hb[ii:]
        self.depthb = self.depthb[ii:]
        self.sl = self.sl[ii:]
        self.Omega = self.Omega[ii:]
        self.wast = self.wast[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.hb_splited = self.hb[idx]
        self.depthb_splited = self.depthb[idx]
        self.sl_splited = self.sl[idx]
        self.Omega_splited = self.Omega[idx]
        self.wast_splited = self.wast[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]

        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)

        self.full_run = self.run_model(self.solution)

        if self.switch_Yini == 0:
            self.par_names = [r'k-', r'k+', r'Y_0']
            self.par_values = self.solution.copy()
            self.par_values[0] = np.exp(self.par_values[0])
            self.par_values[1] = np.exp(self.par_values[1])
        elif self.switch_Yini == 1:
            self.par_names = [r'k-', r'k+', r'Y_0', r'Y_i']
            self.par_values = self.solution.copy()
            self.par_values[0] = np.exp(self.par_values[0])
            self.par_values[1] = np.exp(self.par_values[1])

