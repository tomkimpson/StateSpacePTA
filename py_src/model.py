

import numpy as np

from gravitational_waves import GWs, gw_prefactor,gw_modulation

class LinearModel:

    """
    A linear model of the state evolution
    
    """


    def F_function(gamma,dt):
    
        value = np.exp(-gamma*dt)
        #return np.diag(value)
        return value


    def T_function(f0,fdot,gamma,t,dt):
      
        tensor_product =  np.outer(t+dt,fdot) #This has shape(n times, n pulsars)

        # print("size of tp: ", tensor_product.shape)
        # print("size of f0: ", f0.shape)
        # #print("size of fdot*dt", (fdot*dt).shape)
        # print(f0)
        # print(tensor_product)
        # print("-------------------")

        # print(f0 + tensor_product)
       

        #print("GAmma values = ", gamma)
        #value = f0 + fdot*(t+dt) - np.exp(-gamma*dt)*(f0+fdot*t)
        value = f0 + tensor_product + fdot*dt - np.exp(-gamma*dt)*(f0+tensor_product)

        
        return value
 

    """
    Measurement function which takes the state and returns the measurement
    """
    def H_function(t,omega,phi0,prefactor,dot_product):
        GW_factor = gw_modulation(t, omega,phi0,prefactor,dot_product)
        return np.diag(GW_factor) 
  

    """
    Measurement function which takes the state and returns the measurement
    """
    def H_function_i(modulation_factors):
        GW_factor = modulation_factors #[i,:]
        return GW_factor
  


    """
    Returns a Q matrix 
    """
    def Q_function(gamma,sigma_p,dt):
        
        value = sigma_p**2 * (np.exp(2.0*gamma* dt) - 1.) / (2.0 * gamma)
        #return np.diag(value) 
        return value #this is now a vector
     

    def R_function(L, sigma_m):
        #return np.diag(np.full(L,sigma_m**2)) 
        return sigma_m**2 #this is now a scalar
     

