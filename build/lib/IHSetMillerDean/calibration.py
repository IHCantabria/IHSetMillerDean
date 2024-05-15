import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from .millerDean import millerDean
from IHSetCalibration import objective_functions
from IHSetUtils import wMOORE, wast

class cal_MillerDean(object):
    """
    cal_MillerDean
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wavb.nc')
        ens = xr.open_dataset(path+'ens.nc')
        slv = xr.open_dataset(path+'sl.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = cfg['dt'].values
        self.switch_Yini = cfg['switch_Yini'].values
        self.D50 = cfg['D50'].values
        self.Hberm = cfg['Hberm'].values
        self.flagP = cfg['flagP'].values

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hb = wav['Hb'].values
        self.Tp = wav['Tp'].values
        self.Dirb = wav['Dirb'].values
        self.depthb = wav['depthb'].values

        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
        
        self.Obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        try:
            self.tide = slv['tide'].values
            self.surge = slv['surge'].values
            self.sl = slv['sl'].values
        except:
            self.sl = slv['sl'].values

        self.ws = wMOORE(self.D50)
        self.Omega = self.Hb / (self.ws * self.Tp)
        self.wast = wast(self.depthb, self.D50)
        
        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

        cfg.close()
        wav.close()
        ens.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        
        

        if self.switch_Yini == 0:
            def model_simulation(par):
                kero = par['kero']
                kacr = par['kacr']
                Y0 = par['Y0']
                
                Ymd, _ = millerDean(self.Hb_splited,
                                    self.depthb_splited,
                                    self.sl_splited,
                                    self.wast_splited,
                                    self.dt,
                                    self.Hberm,
                                    Y0,
                                    kero,
                                    kacr,
                                    self.Yini,
                                    self.flagP,
                                    self.Omega_splited)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('kero', 1e-7, 1e-2),
                Uniform('kacr', 1e-7, 1e-2),
                Uniform('Y0', 0.25*np.min(self.Obs), 2*np.max(self.Obs))
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 1:
            def model_simulation(par):
                kero = par['kero']
                kacr = par['kacr']
                Y0 = par['Y0']
                Yini = par['Yini']
                
                Ymd, _ = millerDean(self.Hb_splited,
                                    self.depthb_splited,
                                    self.sl_splited,
                                    self.wast_splited,
                                    self.dt,
                                    self.Hberm,
                                    Y0,
                                    kero,
                                    kacr,
                                    Yini,
                                    self.flagP,
                                    self.Omega_splited)
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('kero', 1e-7, 1e-2),
                Uniform('kacr', 1e-7, 1e-2),
                Uniform('Y0', 0.25*np.min(self.Obs), 2*np.max(self.Obs)),
                Uniform('Yini', np.min(self.Obs), (self.Obs))
            ]
            self.model_sim = model_simulation

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """ 
        ii = np.where(self.time>=self.start_date)[0][0]
        self.Hb = self.Hb[ii:]
        self.depthb = self.depthb[ii:]
        self.sl = self.sl[ii:]
        self.Omega = self.Omega[ii:]
        self.wast = self.wast[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.Hb_splited = self.Hb[idx]
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
        self.idx_validation_obs = idx[0]
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

