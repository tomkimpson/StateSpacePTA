


import bilby

import numpy as np
import logging 
logging.getLogger().setLevel(logging.INFO)




"""
Return the true parameter values
"""
def optimal_parameter_values(pulsar_parameters,P):
    return [P.Ω,P.Φ0,P.ψ,P.ι,P.δ,P.α,P.h]

"""
Return the true parameter values,perturbed by some small amount
"""
def sub_optimal_parameter_values(pulsar_parameters,P):
    return [P.Ω*1.05,P.Φ0*1.05,P.ψ*1.05,P.ι*1.05,P.δ*1.05,P.α*1.05,P.h*1.05]




"""
Set up prior for GW params. See e.g. https://emcee.readthedocs.io/en/stable/tutorials/line/
"""
def log_prior(theta):
    omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h = theta
    uniform_condition = (1e-7 < omega_gw < 1e-6) and (0.1 < phi0_gw < 0.3) and (2.0 < psi_gw < 3.0) and (0.5 < iota_gw < 1.5) and (0.5 < delta_gw < 1.5) and (0.5 < alpha_gw < 1.5) and (0.5e-12 < h < 1.5e-12)
    if uniform_condition:
        return 0.0
    return -np.inf



def log_probability(theta, KF):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + KF.likelihood(theta)

   