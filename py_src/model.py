

import numpy as np
from numba import jit 
#from gravitational_waves import GWs, gw_prefactor,gw_modulation

class LinearModel:

    """
    A linear model of the state evolution
    
    """


    @jit(nopython=True)
    def F_function(gamma,dt):
    
        value = np.exp(-gamma*dt)
        return value


    @jit(nopython=True)
    def T_function(f0,fdot,gamma,t,dt):
      
        tensor_product =  np.outer(t+dt,fdot) #This has shape(n times, n pulsars)

        value = f0 + tensor_product + fdot*dt - np.exp(-gamma*dt)*(f0+tensor_product)
        
        return value
 

    # """
    # Measurement function which takes the state and returns the measurement
    # """
    # def H_function(t,omega,phi0,prefactor,dot_product):
    #     GW_factor = gw_modulation(t, omega,phi0,prefactor,dot_product)
    #     return np.diag(GW_factor) 
  

    """
    Measurement function which takes the state and returns the measurement
    """
    @jit(nopython=True)
    def H_function_i(modulation_factors):
        GW_factor = modulation_factors #[i,:]
        return GW_factor
  


    """
    Returns a Q matrix 
    """
    @jit(nopython=True)
    def Q_function(gamma,sigma_p,dt):

        print("The input sigma to the Q function = ",sigma_p )
        value = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
        return value #this is now a vector
     

    @jit(nopython=True)
    def R_function(L, sigma_m):
        #return np.diag(np.full(L,sigma_m**2)) 
        return sigma_m**2 #this is now a scalar
     

