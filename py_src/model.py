

import numpy as np
from numba import jit 
class LinearModel:

    """
    A linear model of the state evolution
    
    """

    """
    The diagonal F matrix as a vector
    """
    @jit(nopython=True)
    def F_function(gamma,dt):
        return np.exp(-gamma*dt)


    """
    The control vector
    """
    @jit(nopython=True)
    def T_function(f0,fdot,gamma,t,dt):

       
        fdot_time =  np.outer(t+dt,fdot) #This has shape(n times, n pulsars)
        value = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)

        return value


    """
    The diagonal Q matrix as a vector
    """
    @jit(nopython=True)
    def Q_function(gamma,sigma_p,dt):
        return -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
     

    """
    The R matrix as a scalar
    """
    @jit(nopython=True)
    def R_function(sigma_m):
        return sigma_m**2
     

