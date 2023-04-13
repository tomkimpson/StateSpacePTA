# Python specific imports that will make our job easier and our code prettier
from collections import namedtuple
from functools import partial
import time
from tqdm.auto import trange, tqdm

# JAX specific imports that we will use to code the logic
from jax import jit, vmap, make_jaxpr, device_put, devices
from jax.config import config
from jax.core import eval_jaxpr  # this will serve as a proxy compiler as JAX doesn't have an AOT function
from jax.lax import associative_scan, psum, scan
import jax.numpy as jnp
import jax.scipy as jsc

# Auxiliary libraries that we will use to report results and create the data
import math
import matplotlib.pyplot as plt
import jax.numpy as np


from jaxns import Prior, Model
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions


from jax import random
from jax import lax

from gravitational_waves import gw_prefactor_optimised
from jax.scipy.stats import multivariate_normal



config.update("jax_enable_x64", True)  # We use this so that we have the same precision as the pure numpy implementation
                                       # This can be useful in particular for large observations (long running series)


LOG2PI = math.log(2 * math.pi)




from system_parameters import SystemParameters,SystemParams
from pulsars import Pulsars
from synthetic_data import SyntheticData

P   = SystemParameters()       #define the system parameters as a class
params = SystemParams()
PTA = Pulsars(P)               #setup the PTA
data = SyntheticData(PTA,P)    # generate some synthetic data
y = data.f_measured












def kalman_filter(y, F, Q, R, x0, P1, H_fn,T_fn):




    def body(carry, t):
        x_hat_tm1, P_tm1,log_likelihood = carry

        #Get the measurement matrix, the control vector matrix and the observation obs matrix
        H_t = H_fn[t]
        T_t = T_fn[t]
        y_t = y[t]

        # Predict state estimate and error covariance
        x_hat_t = F * x_hat_tm1 + T_t
        P_t = F * P_tm1 * F.T + Q

        # Compute Kalman gain
        S = H_t * P_t * H_t + R
        K_t = P_t * H_t/S

       
        # Update state estimate and error covariance
        innovation = y_t - H_t * x_hat_t

        x_hat = x_hat_t + K_t * innovation
        #print("xhat gain shape:", x_hat.shape)

        I_KH = 1.0 - K_t*H_t
        P = I_KH * P_t * I_KH + K_t * R * K_t

        # Calculate log likelihood

        x = innovation / S 
        N = len(x)
        log_likelihood = -0.5*(np.dot(innovation,x) + N*np.log(2*np.pi))

        #P = (jnp.eye(n_dim) - K_t @ H_t) @ P_t

        return (x_hat, P,log_likelihood), (x_hat, P,log_likelihood)

    n_obs, n_dim = y.shape

   

    # Initialize state estimates
    x_hat0 = jnp.zeros((n_dim,))
    x_hat0 = x_hat0.at[...].set(x0)

   
    
    P0 = jnp.zeros((n_dim,))
    P0 = P0.at[...].set(P1)
    


    # Initialize log likelihood
    #log_likelihood = jnp.float64(0.)


    #perform a single update step
    #ti = t[0]
    H_t = H_fn[0]
    S = H_t * P0 * H_t + R
    K_t = P0 * H_t/S

    #Update state estimate and error covariance
    y_t = y[0]
    innovation = y_t - H_t * x_hat0

    x_hat = x_hat0 + K_t * innovation
    print(x_hat0,  K_t, innovation)
        

    I_KH = 1.0 - K_t*H_t
    P = I_KH * P0 * I_KH + K_t * R * K_t

    # Calculate log likelihood

    x = innovation / S 
    N = len(x)
    ll = -0.5*(np.dot(innovation,x) + N*np.log(2*np.pi))



    log_likelihood = jnp.float64(ll)

    x_hat0 = x_hat0.at[...].set(x_hat)
    P0 = P0.at[...].set(P)
    


    # Iterate over observations using scan
    _, (x_hat, P,log_likelihood) = lax.scan(body, (x_hat0, P0,log_likelihood), jnp.arange(1, n_obs))


    # Prepend initial state estimate and error covariance
    x_hat = jnp.concatenate((x_hat0[jnp.newaxis, :], x_hat), axis=0)
    P = jnp.concatenate((P0[jnp.newaxis, :], P), axis=0)

    return x_hat, P,log_likelihood






def prior_model():
    omega = yield Prior(tfpd.Uniform(low=4.5e-7, high = 5.5e-7), name='omega')
    return omega




def log_likelihood(omega): 

    
    #Define some parameters from the global scope by hand
    gamma = PTA.gamma
    dt = PTA.dt
    f0 = PTA.f
    fdot = PTA.fdot
    t = PTA.t
    sigma_p= PTA.sigma_p
    sigma_m= PTA.sigma_m


    #omega_gw  =P["omega_gw"]
    #omega_gw = 4e-7
    phi0_gw  =params.phi0_gw
    psi_gw   =params.psi_gw
    iota_gw  =params.iota_gw
    delta_gw =params.delta_gw
    alpha_gw =params.alpha_gw
    h        =params.h
    d = PTA.d



    F = np.exp(-gamma*dt)


    fdot_time =  np.outer(t,fdot) #This has shape(n times, n pulsars)
    T_fn = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)


    Q = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
    R = sigma_m**2


    x0 = y[0,:]
    P0 = np.ones(len(x0)) * sigma_m*1e10 



    H_fn = gw_prefactor_optimised(delta_gw,
                                                    alpha_gw,
                                                    psi_gw,
                                                    PTA.q,
                                                    PTA.q_products,
                                                    h,
                                                    iota_gw,
                                                    omega,
                                                    d,
                                                    t,
                                                    phi0_gw
                                                    )


    x,P,l = kalman_filter(y, F, Q, R, x0, P0, H_fn,T_fn)


    value = np.sum(l)
    print("Returned likelihood is: ", value)

    return value



#value = log_likelihood(4e-7)
#print(value.dtype)


model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
print("Perform a sanity check")
model.sanity_check(random.PRNGKey(0), S=100)


from jaxns import ExactNestedSampler
from jaxns import TerminationCondition

print("--------------Attepting to run nested sampler-----------------")
# Create the nested sampler class. In this case without any tuning.
ns = exact_ns = ExactNestedSampler(model=model, num_live_points=100, num_parallel_samplers=1,
                                   max_samples=1e4)

termination_reason, state = exact_ns(random.PRNGKey(42),
                                     term_cond=TerminationCondition(live_evidence_frac=1e-2))
results = exact_ns.to_results(state, termination_reason)

print("COMPLETED")
print(exact_ns.summary(results))

exact_ns.plot_cornerplot(results)






 




#print("likelihood shape = ", np.sum(l))


from plotting import plot_all



#plot_all(PTA.t,data.intrinsic_frequency,data.f_measured,x,psr_index =1,savefig=None)








# fig, ax = plt.subplots(figsize=(7, 7))
# psr_index = 1
# ax.plot(PTA.t, y[:,psr_index], label="Measured State")

# #plt.show()
# #ax.plot(fms[:100, 0], fms[:100, 1], label="Filtered", color="g", linestyle="--")
# #ax.plot(sms[:100, 0], sms[:100, 1], label="Smoothed", color="k", linestyle="--")
# #ax.plot(FSms[:100, 0], FSPs[:100, 1], label="Filter-Smoothed", color="k", linestyle="--")

# #a#x.scatter(*ys[:100].T, label="Observations", color="r")
# _ = plt.legend()
# plt.show()





#Generate observations
#log10T = 4
#true_xs, ys = get_data(car_tracking_model, 10 ** log10T, 0)

























# #Kalman filter























#print(fms.shape)
#print(lPs.shape)
# #Kalman smoother
# sms, sPs = ks(car_tracking_model, fms, fPs)


# #filter-smoother
# def kfs(model, observations):
#     return ks(model, *kf(model, observations))


# FSms, FSPs = kfs(car_tracking_model, ys[:100])











