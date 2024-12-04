import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .millerDean import millerDean_njit
import json
from IHSetUtils import wMOORE, wast, BreakingPropagation
from scipy.stats import circmean


class Yates09_run(object):
    """
    Yates09_run
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['run_MillerDean'])

        self.D50 = cfg['D50']
        self.hberm = cfg['Hberm']
        self.depth = cfg['depth']
        self.flagP = cfg['flagP']

        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.time = pd.to_datetime(data.time.values)
            self.E = self.hs ** 2
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.E = self.hs ** 2
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]

        if cfg['switch_Yini'] == 1:
            ii = np.argmin(np.abs(self.time_obs - self.time[0]))
            self.Yini = self.Obs[ii]
        
        data.close()

        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])

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
            def run_model(par):
                kero = np.exp(par[0])
                kacr = np.exp(par[1])
                Y0 = par[2]
                
                Ymd, _ = millerDean_njit(self.hb,
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

        elif self.switch_Yini == 1:
            def run_model(par):
                kero = np.exp(par[0])
                kacr = np.exp(par[1])
                Y0 = par[2]
                Yini = par[3]
                
                Ymd, _ = millerDean_njit(self.hb,
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
    
    def run(self, par):
        self.full_run = self.run_model(par)
        self.calculate_metrics()

    def calculate_metrics(self):
        self.metrics_names = fo.backtot()[0]
        self.indexes = fo.multi_obj_indexes(self.metrics_names)
        self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """ 
        ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.hb = self.hb[ii]
        self.depthb = self.depthb[ii]
        self.sl = self.sl[ii]
        self.Omega = self.Omega[ii]
        self.wast = self.wast[ii]
        self.time = self.time[ii]



