

import numpy as np
from numba import jit,config

from system_parameters import disable_JIT
config.DISABLE_JIT = disable_JIT

class LinearModel:

    """
    A linear model of the state evolution
    
    """

    """
    The diagonal F matrix as a vector
    """
    #@jit(nopython=True)
    def F_function(gamma,dt):
        return np.exp(-gamma*dt)


    """
    The control vector
    """
    #@jit(nopython=True)
    def T_function(f0,fdot,gamma,t,dt):

       
        #fdot_time =  np.outer(t+dt,fdot) #This has shape(n times, n pulsars)
        #value = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)
        
        
        fdot_time =  np.outer(t,fdot) #This has shape(n times, n pulsars)
        value = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)

        return value #/f0


    """
    The diagonal Q matrix as a vector
    """
    #@jit(nopython=True)
    def Q_function(gamma,sigma_p,dt,f0):
        value = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
        return value #/f0**2
     

    """
    The R matrix as a scalar
    """
    #@jit(nopython=True)
    def R_function(sigma_m,f0):
        return sigma_m**2 #/f0**2
     

