

import numpy as np
from numba import jit,config

from system_parameters import disable_JIT,scaling_factor_x
config.DISABLE_JIT = disable_JIT
from gravitational_waves import gw_psr_terms,gw_earth_terms

scaling_factor = scaling_factor_x

class LinearModel:

    """
    A linear model of the state evolution
    
    """

    def __init__(self,P):

        """
        Initialize the class. 
        """

        if P["psr_terms_model"]:
            print("Attention: You are including the PSR terms in your measurement model")
            self.H_function = gw_psr_terms
        else:
            print("Attention: You are using just the Earth terms in your measurement model")
            self.H_function = gw_earth_terms




#These functions have to be outside the class to enable JIT compilation
#Bit ugly, but works from a performance standpoint
#To do: clean up
"""
The diagonal F matrix as a vector
"""
@jit(nopython=True)
def F_function(gamma,dt):
    return np.exp(-gamma*scaling_factor*dt)


"""
The control vector
"""
@jit(nopython=True)
def T_function(f0,fdot,gamma,t,dt):

    
    fdot_time =  np.outer(t,fdot) #This has shape(n times, n pulsars)
    value = f0 + fdot_time + fdot*dt - np.exp(-gamma*scaling_factor*dt)*(f0+fdot_time)

    return value*scaling_factor


"""
The diagonal Q matrix as a vector
"""
@jit(nopython=True)
def Q_function(gamma,sigma_p,dt):
    value = -(sigma_p*scaling_factor)**2 * (np.exp(-2.0*gamma*scaling_factor*dt) - 1.) / (2.0 * gamma*scaling_factor)

    #print("The value of the q function is:", value)
    return value 
    

"""
The R matrix as a scalar
"""
@jit(nopython=True)
def R_function(sigma_m):
    return (sigma_m*scaling_factor)**2 
    

