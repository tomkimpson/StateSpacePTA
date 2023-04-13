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
import numpy as np
import scipy as sc


from jaxns import Prior, Model
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions


from jax import random




config.update("jax_enable_x64", True)  # We use this so that we have the same precision as the pure numpy implementation
                                       # This can be useful in particular for large observations (long running series)


LOG2PI = math.log(2 * math.pi)



StateSpaceModel = namedtuple("StateSpaceModel", ["F", "H", "Q", "R", "m0", "P0", "xdim", "ydim","likelihood"])


def make_car_tracking_model(q: float, dt: float, r: float, m0: np.ndarray, P0: np.ndarray):
    F = np.eye(4) + dt * np.eye(4, k=2)
    H = np.eye(2, 4)
    Q = np.kron(np.array([[dt**3/3, dt**2/2],
                          [dt**2/2, dt]]), 
                np.eye(2))
    R = r ** 2 * np.eye(2)
    return StateSpaceModel(F, H, q * Q, R, m0, P0, m0.shape[0], H.shape[0],0.0)


car_tracking_model = make_car_tracking_model(q=1., dt=0.1, r=0.5, 
                                             m0=np.array([0., 0., 1., -1.]), 
                                             P0=np.eye(4)
                                             )





def get_data(model: StateSpaceModel, T:float, seed:int=0):
    # We first generate the normals we will be using to simulate the SSM:
    rng = np.random.RandomState(seed)
    normals = rng.randn(1 + T, model.xdim + model.ydim)
    
    # Then we allocate the arrays where the simulated path and observations will
    # be stored:
    xs = np.empty((T, model.xdim))
    ys = np.empty((T, model.ydim))

    # So that we can now run the sampling routine:
    Q_chol = sc.linalg.cholesky(model.Q, lower=True)
    R_chol = sc.linalg.cholesky(model.R, lower=True)
    P0_chol = sc.linalg.cholesky(model.P0, lower=True)
    x = model.m0 + P0_chol @ normals[0, :model.xdim]
    for i, norm in enumerate(normals[1:]):
        x = model.F @ x + Q_chol @ norm[:model.xdim]
        y = model.H @ x + R_chol @ norm[model.xdim:]
        xs[i] = x
        ys[i] = y
    return xs, ys


def mvn_logpdf(x, mean, cov):
    n = mean.shape[0]
    upper = jsc.linalg.cholesky(cov, lower=False)
    log_det = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(upper))))
    diff = x - mean
    scaled_diff = jsc.linalg.solve_triangular(upper, diff.T, lower=False)
    distance = jnp.sum(scaled_diff * scaled_diff, 0)
    return -0.5 * (distance + n * LOG2PI + log_det)



def kf(model, observations):
    def body(carry, y):
        m, P,l = carry
        m = model.F @ m
        P = model.F @ P @ model.F.T + model.Q

        obs_mean = model.H @ m
        S = model.H @ P @ model.H.T + model.R

        K = jsc.linalg.solve(S, model.H @ P, sym_pos=True).T  # notice the jsc here
        m = m + K @ (y - model.H @ m)
        P = P - K @ S @ K.T

        l = mvn_logpdf(y,obs_mean,S)
        return (m, P,l), (m, P,l)

    _, (fms, fPs,fls) = scan(body, (model.m0, model.P0,model.likelihood), observations)
    return fms, fPs,fls



# def ks(model, ms, Ps):
#     def body(carry, inp):
#         m, P = inp
#         sm, sP = carry

#         pm = model.F @ m
#         pP = model.F @ P @ model.F.T + model.Q

#         C = jsc.linalg.solve(pP, model.F @ P, sym_pos=True).T  # notice the jsc here
        
#         sm = m + C @ (sm - pm)
#         sP = P + C @ (sP - pP) @ C.T
#         return (sm, sP), (sm, sP)

#     _, (sms, sPs) = scan(body, (ms[-1], Ps[-1]), (ms[:-1], Ps[:-1]), reverse=True)
#     sms = jnp.append(sms, jnp.expand_dims(ms[-1], 0), 0)
#     sPs = jnp.append(sPs, jnp.expand_dims(Ps[-1], 0), 0)
#     return sms, sPs



#Generate observations
log10T = 4
true_xs, ys = get_data(car_tracking_model, 10 ** log10T, 0)






#Kalman filter


def prior_model():
    omega = yield Prior(tfpd.Uniform(low=3e-7, high = 7e-7), name='omega')
    return omega








def log_likelihood(omega): 

    fms, fPs,lPs = kf(car_tracking_model, ys[:100])


    return jnp.sum(lPs)








model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
print("Perform a sanity check")
model.sanity_check(random.PRNGKey(0), S=100)














#print(fms.shape)
#print(lPs.shape)
# #Kalman smoother
# sms, sPs = ks(car_tracking_model, fms, fPs)


# #filter-smoother
# def kfs(model, observations):
#     return ks(model, *kf(model, observations))


# FSms, FSPs = kfs(car_tracking_model, ys[:100])












fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(true_xs[:100, 0], true_xs[:100, 1], label="True State", color="b")
ax.plot(fms[:100, 0], fms[:100, 1], label="Filtered", color="g", linestyle="--")
#ax.plot(sms[:100, 0], sms[:100, 1], label="Smoothed", color="k", linestyle="--")
#ax.plot(FSms[:100, 0], FSPs[:100, 1], label="Filter-Smoothed", color="k", linestyle="--")

ax.scatter(*ys[:100].T, label="Observations", color="r")
_ = plt.legend()
plt.show()