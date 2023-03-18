

import numpy as np

from gravitational_waves import GWs, gw_prefactor,gw_modulation

class LinearModel:

    """
    A linear model of the state evolution
    
    """

    def __init__(self,dt,Npulsars):
 
        self.dt = dt 
        self.Npulsars = Npulsars
        self.dims_x   = Npulsars 
        self.dims_z   = Npulsars


    def F_function(gamma,dt):
    
        value = np.exp(-gamma*dt)
        return np.diag(value)


    def T_function(gamma,f0,fdot,t,dt):
  
        value = f0 + fdot*(t+dt) - np.exp(-gamma*dt)*(f0+fdot*t)
        return value
 

    """
    Measurement function which takes the state and returns the measurement
    """
    def H_function(P,t,q,d):

        GW = GWs(P)  
        prefactor, dot_product =gw_prefactor(GW.n,q, GW.Hij, GW.omega_gw, d)
        GW_factor = gw_modulation(t, GW.omega_gw,GW.phi0_gw,prefactor,dot_product)
        return np.diag(GW_factor) 
  

    """
    Returns a Q matrix 
    """
    def Q_function(gamma,sigma_p,dt):
        value = sigma_p**2 * (np.exp(2.0*gamma* dt) - 1.0 / (2.0 * gamma))
        return np.diag(value) 
     

    def R_function(L, sigma_m):
        return np.diag(np.full(L,sigma_m**2)) 
     

