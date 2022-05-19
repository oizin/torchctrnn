import numpy as np
import pandas as pd
import random
import math
from numba import jit
import typing

# globals
DT = 1e-2

# Ornstein Uhlenbeck ----------------------------------------------

@jit(nopython=True)
def _OE_simulate_trajectory():

    dt = 0.01
    maxtime = 24.0
    n_iter = int(maxtime / dt) + 1
    saveat = range(0,n_iter,int(0.1 / dt))

    # output container
    column_name_pos = {'t':0,
                       'y_t':1,
                       'obs':2}
    output = np.zeros((len(saveat), len(column_name_pos)))
    output[:,column_name_pos['t']] = np.linspace(0,maxtime,len(saveat))

    # set parameters
    sigma = np.random.lognormal()
    mu = np.random.normal()
    theta = np.random.normal(0.5,0.01)
    
    # time 0
    y_t = np.random.normal()
    iter = 0
    save_step = 0
    iter_measure = 0
    obs = False

    ## loop through time steps
    while iter <= n_iter:
        if iter == iter_measure:
            obs = True
            iter_measure = min(iter + np.round(np.random.poisson(200),-1),n_iter)
        else:
            obs = False

        dW = np.random.normal(0,dt)
        d_y_deter = (theta*(mu - y_t))*dt 
        d_y_stoch = np.sqrt(2*theta*sigma**2)*dW
        d_y = d_y_deter + d_y_stoch
        y_t = y_t + d_y

        if iter in saveat:
            output[save_step,column_name_pos['y_t']] = y_t
            output[save_step,column_name_pos['obs']] = obs
            save_step += 1
        iter += 1

    return output

@jit(nopython=True)
def _OE_simulate(N:int,seed:int=None):

    np.random.seed(seed)

    # output container
    n_iter = int(24.0 / 1e-2) + 1
    saveat = range(0,n_iter,int(0.1 / 1e-2))
    n_save_steps = len(saveat)
    output = np.zeros((n_save_steps*N, 3))

    for i in range(N):
        output_i = _OE_simulate_trajectory()
        lower = i*n_save_steps
        upper = (i+1)*n_save_steps
        output[lower:upper,:] = output_i

    return output

class OrnsteinUhlenbeckData:
    
    def __init__(self,seed=None):
        """
        Args:
            seed
        """
        # arguments
        self.seed = seed

        # other
        self.dt = 0.01
        self.maxtime = 24.0
        self.columns = ['t','y_t','obs']

    def simulate(self,N:int=100): # TODO: the seed should be here!

        # output container
        n_iter = int(self.maxtime / self.dt) + 1
        saveat = range(0,n_iter,int(0.1 / self.dt))
        n_save_steps = len(saveat)

        output = _OE_simulate(N,self.seed)

        df = pd.DataFrame(output,columns=self.columns)
        df['id'] = np.repeat(range(0,N),n_save_steps)
        df['obs'] = (df.obs == 1.0)
        return df

# Glucose data ----------------------------------------------------

@jit(nopython=True)
def _Gluc_temporal_process(glucose:float,insulin_dose:float):
    """
    Next observation time
    """
    # insulin being used?
    if insulin_dose > 0:
        base = 3.0
    else:
        base = 5.0
    # measured glucose
    if glucose < 80.0:
        mean = base * np.exp(-((glucose - 120.0)/50.0) ** 2.0)
    else: 
        mean = base * np.exp(-((glucose - 120.0)/140.0) ** 2.0)
    # some randomness in times
    log_time = np.random.normal(np.log(mean),0.1)
    time = np.exp(log_time)
    return time

@jit(nopython=True)
def _Gluc_insulin_policy(glucose:float,insulin_dose:float=0.0):
    """
    dose over an hour
    """
    if (glucose > 0) and (glucose < 140):
        dose_hr = 0.0
    elif glucose >= 140 and glucose < 160:
        dose_hr = 3.0
    elif glucose >= 160 and glucose < 200:
        dose_hr = 20.0
    else:
        dose_hr = 30.0
    return dose_hr / 60.0

@jit(nopython=True)
def _Gluc_dextrose_policy(t:float):
    """
    """
    if np.random.random() > 0.9:
        glucose_mg_min = 180.0 + np.random.random()*10.0
    else: 
        glucose_mg_min = 0.0
    return glucose_mg_min / 50.0

@jit(nopython=True)
def _Gluc_simulate_trajectory(sigma_m:float=0.0):

    dt = 0.01
    maxtime = 24.0
    n_iter = int(maxtime / dt) + 1
    saveat = range(0,n_iter,int(0.1 / dt))

    # output container
    column_name_pos = {'t':0,
                       'glucose_t':1,
                       'glucose_t_obs':2,
                       'obs':3,
                       'insulin_t':4,
                       'dextrose_t':5}
    output = np.zeros((len(saveat), len(column_name_pos)))
    output[:,column_name_pos['t']] = np.linspace(0,maxtime,len(saveat))

    # set parameters
    beta = -np.random.normal(50,5)
    sigma = np.random.normal(20.0,2)
    mu = np.random.normal(140,5)
    theta = np.random.normal(0.5,0.01)
    
    # time 0
    insulin_t = 0.0
    glucose_t = np.random.normal(140,20)
    glucose_t_obs = np.exp(np.random.normal(np.log(glucose_t),sigma_m))
    iter = 0
    save_step = 0
    iter_measure = 0
    obs = False

    ## loop through time steps
    while iter <= n_iter:
        if iter == iter_measure:
            obs = True
            dextrose_t = _Gluc_dextrose_policy(iter * dt)
            insulin_t = _Gluc_insulin_policy(glucose_t,insulin_t)
            delta_t = _Gluc_temporal_process(glucose_t,insulin_t)
            iter_measure = min(iter + int(np.round(delta_t/dt,-1)),n_iter)
            
        else:
            obs = False

        dW = np.random.normal(0,dt)
        d_glucose_deter = (theta*(mu - glucose_t) + beta*insulin_t + dextrose_t)*dt 
        d_glucose_stoch = np.sqrt(2*theta*sigma**2)*dW
        d_glucose = d_glucose_deter + d_glucose_stoch
        glucose_t = glucose_t + d_glucose
        glucose_t_obs = np.exp(np.random.normal(np.log(glucose_t),sigma_m))

        if iter in saveat:
            output[save_step,column_name_pos['glucose_t']] = glucose_t
            output[save_step,column_name_pos['glucose_t_obs']] = glucose_t_obs
            output[save_step,column_name_pos['obs']] = obs
            output[save_step,column_name_pos['insulin_t']] = insulin_t
            output[save_step,column_name_pos['dextrose_t']] = dextrose_t 
            save_step += 1
        iter += 1

    return output

@jit(nopython=True)
def _Gluc_simulate(N:int,sigm_m:float=0.0,seed:int=None):

    np.random.seed(seed)

    # output container
    n_iter = int(24.0 / 1e-2) + 1
    saveat = range(0,n_iter,int(0.1 / 1e-2))
    n_save_steps = len(saveat)
    output = np.zeros((n_save_steps*N, 6))

    for i in range(N):
        output_i = _Gluc_simulate_trajectory(sigm_m)
        lower = i*n_save_steps
        upper = (i+1)*n_save_steps
        output[lower:upper,:] = output_i

    return output

class GlucoseData:
    
    def __init__(self,measurement_error:float=0.0,seed=None):
        """
        Args:
            measurement_error
        """
        # arguments
        self.sigma_m = measurement_error
        self.seed = seed

        # other
        self.dt = 0.01
        self.maxtime = 24.0
        self.columns = ['t','glucose_t','glucose_t_obs',
                       'obs','insulin_t','dextrose_t']

    def simulate(self,N:int=100):

        # output container
        n_iter = int(self.maxtime / self.dt) + 1
        saveat = range(0,n_iter,int(0.1 / self.dt))
        n_save_steps = len(saveat)

        output = _Gluc_simulate(N,self.sigma_m,self.seed)

        df = pd.DataFrame(output,columns=self.columns)
        df['id'] = np.repeat(range(0,N),n_save_steps)
        df['obs'] = (df.obs == 1.0)
        return df