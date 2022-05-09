from scipy.stats import norm,lognorm 
import numpy as np
import pandas as pd
import random
from typing import Callable

def next_observation_time(glucose,insulin_dose):
    """
    Next observation time
    """
    # insulin being used?
    if insulin_dose > 0:
        base = 3
    else:
        base = 5
    # measured glucose
    if glucose < 80:
        mean = base * np.exp(-((glucose - 120)/50) ** 2)
    else: 
        mean = base * np.exp(-((glucose - 120)/140) ** 2)
    # some randomness in times
    log_time = norm.rvs(loc=np.log(mean),scale=0.1)
    time = np.exp(log_time)
    return time

def insulin_input(glucose):
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

def dextrose_input(t):

    if random.random() > 0.9:
        glucose_mg_min = 180.0 + round(random.random()*10) 
    else: 
        glucose_mg_min = 0.0
    return glucose_mg_min / 50.0


class GlucoseData:
    
    def __init__(self,
                insulin_policy:Callable=insulin_input,
                dextrose_policy:Callable=dextrose_input,
                temporal_process:Callable=next_observation_time,
                measurement_error:float=0.0):
        """
        Args:
            insulin_policy
            dextrose_policy
            temporal_process
            measurement_error
        """
        # arguments
        self.insulin_policy = insulin_policy
        self.dextrose_policy = dextrose_policy
        self.temporal_process = temporal_process
        self.sigma_m = measurement_error

        # 
        self.dt = 1e-2
        self.maxtime = 24.0


    def simulate_trajectory(self):

        # random variables
        p_theta = norm(0.5,0.01)  # mean reversion strength of process
        p_sigma = norm(20.0,2)  # variance of process
        p_mu = norm(140,5)  # mean of process
        p_t0 = norm(140,20)  # initial value
        p_beta = norm(50,5)  # insulin effect

        # time
        dt = self.dt
        maxtime = self.maxtime
        n_iter = int(maxtime / dt) + 1
        saveat = range(0,n_iter,int(0.1 / dt))

        # output container
        output = pd.DataFrame(np.zeros((len(saveat), 5)),
                                columns=['glucose_t','glucose_t_obs',
                                        'obs','insulin_t','dextrose_t'])
        output['t'] = np.linspace(0,maxtime,len(saveat))

        # set parameters
        beta = -p_beta.rvs(1)
        sigma = p_sigma.rvs(1)
        mu = p_mu.rvs(1)
        theta = p_theta.rvs(1)
        
        # time 0
        glucose_t = glucose_t0 = p_t0.rvs(1)
        glucose_t_obs = glucose_t0  + self.sigma_m * norm.rvs(1) * np.log(glucose_t0)
        iter = 0
        save_step = 0
        iter_measure = 0
        obs = False

        ## loop through time steps
        while iter <= n_iter:
            if iter == iter_measure:
                obs = True
                dextrose_t = self.dextrose_policy(iter * 1e-2)
                insulin_t = self.insulin_policy(glucose_t)
                delta_t = self.temporal_process(glucose_t,insulin_t)
                iter_measure = min(iter + int(round(delta_t/dt,-1)),n_iter)
            else:
                obs = False

            dW = norm.rvs(0,dt,1)
            d_glucose_deter = (theta*(mu - glucose_t) + beta*insulin_t + dextrose_t)*dt 
            d_glucose_stoch = np.sqrt(2*theta*sigma**2)*dW
            d_glucose = d_glucose_deter + d_glucose_stoch
            glucose_t = glucose_t + d_glucose
            glucose_t_obs = glucose_t + self.sigma_m * norm.rvs(size=1) * np.log(glucose_t)

            if iter in saveat:
                output.loc[save_step,'glucose_t'] = glucose_t
                output.loc[save_step,'glucose_t_obs'] = glucose_t_obs
                output.loc[save_step,'obs'] = obs
                output.loc[save_step,'insulin_t'] = insulin_t
                output.loc[save_step,'dextrose_t'] = dextrose_t 
                save_step += 1
            iter += 1

        return output

    def simulate(self,N:int=100):

        # output container
        n_iter = int(self.maxtime / self.dt) + 1
        saveat = range(0,n_iter,int(0.1 / self.dt))
        n_save_steps = len(saveat)
        output = pd.DataFrame(np.zeros((n_save_steps*N, 5)),
                                columns=['glucose_t','glucose_t_obs',
                                        'obs','insulin_t','dextrose_t'])
        output['t'] = np.repeat(np.linspace(0,self.maxtime,n_save_steps),N)

        for i in range(N):
            df_i = self.simulate_trajectory()
            lower = i*n_save_steps
            upper = (i+1)*n_save_steps
            #output.iloc[lower:upper,:] = df_i
            output.loc[lower:upper,'glucose_t'] = df_i.glucose_t
            output.loc[lower:upper,'glucose_t_obs'] = df_i.glucose_t_obs
            output.loc[lower:upper,'obs'] = df_i.obs
            output.loc[lower:upper,'insulin_t'] = df_i.insulin_t
            output.loc[lower:upper,'dextrose_t'] = df_i.dextrose_t 


        return output



# function simulate_ensemble(N::Int,error)
#     # generate
#     df_sim = DataFrame(id=Int64[],
#                         t=Float64[],
#                         obs=Int64[],
#                         x_true=Float64[],
#                         x=Float64[],
#                         m=Float64[],
#                         g=Float64[])
#     for i in 1:N
#         df_i = simulate_glucose_icu(g,next_obs,error);
#         df_i.id = repeat([i],size(df_i,1))
#         append!(df_sim,df_i)
#     end
#     df_sim            
# end