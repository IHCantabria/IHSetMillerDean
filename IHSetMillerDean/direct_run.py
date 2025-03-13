import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .millerDean import millerDean_njit
import json
from IHSetUtils import wMOORE, wast, BreakingPropagation
from scipy.stats import circmean


class MillerDean_run(object):
    """
    run_MillerDean
    
    Configuration to calibrate and run the Miller and Dean (2004) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        self.name = 'Miller and Dean (2004)'

        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['run_MillerDean'])

        self.D50 = cfg['D50']
        self.hberm = cfg['Hberm']
        self.depth = cfg['depth']
        self.flagP = cfg['flagP']
        self.switch_Yini = cfg['switch_Yini']
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.bathy_angle = cfg['bathy_angle']
            self.breakType = cfg['break_type']

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
        
        data.close()

        if self.switch_Yini == 1:
            ii = np.argmin(np.abs(self.time_obs - self.time[0]))
            self.Yini = self.Obs[ii]

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

        if self.switch_Yini == 1:
            ii = np.argmin(np.abs(self.time_obs - self.time[0]))
            self.Yini = self.Obs[ii]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))

        if self.switch_Yini == 0:
            def run_model(par):
                kero = par[0]
                kacr = par[1]
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

        elif self.switch_Yini == 1:
            def run_model(par):
                kero = par[0]
                kacr = par[1]
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
    
    def run(self, par):
        self.full_run = self.run_model(par)
        if self.switch_Yini == 1:
            self.par_names = ['kero', 'kacr', 'Y0']
            self.par_values = self.solution
        elif self.switch_Yini == 0:
            self.par_names = ['kero', 'kacr', 'Y0', 'Yini']
            self.par_values = self.solution

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

        ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs = self.Obs[ii]
        self.time_obs = self.time_obs[ii]



