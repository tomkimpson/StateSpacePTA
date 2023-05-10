

import numpy as np
from numba import jit,config

from system_parameters import disable_JIT
config.DISABLE_JIT = disable_JIT
from gravitational_waves import gw_psr_terms,gw_earth_terms,null_model

class LinearModel:

    """
    A linear model of the state evolution
    
    """

    def __init__(self,P):

        """
        Initialize the class. 
        """


        if P["measurement_model"] == "null":
            print("Attention: You are using just the null measurement model")
            self.H_function = null_model 
        elif P["measurement_model"] == "earth":
            print("Attention: You are using the Earth terms measurement model")
            self.H_function = gw_earth_terms
        elif P["measurement_model"] == "pulsar":
            self.H_function = gw_psr_terms
        else:
            sys.exit("Measurement model not recognized. Stopping.")



#These functions have to be outside the class to enable JIT compilation
#Bit ugly, but works from a performance standpoint
#To do: clean up
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

    
    fdot_time =  np.outer(t,fdot) #This has shape(n times, n pulsars)
    value = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)

    return value 


"""
The diagonal Q matrix as a vector
"""
@jit(nopython=True)
def Q_function(gamma,sigma_p,dt):
    value = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
    return value 
    

"""
The R matrix as a scalar
"""
@jit(nopython=True)
def R_function(sigma_m):
    return sigma_m**2
    
