

import numpy as np 



class SystemParameters:


    def __init__(self):

        #Observation parameters
        self.NF = np.longdouble
        self.T = 10.0  # how long to integrate for in years
        self.cadence = 7.0   # sampling interval in days

        #GW parameters
        self.omega_gw=1e-7      
        self.phi0_gw= 0.20     
        self.psi_gw=  2.5
        self.iota_gw= 0.0
        self.delta_gw= 0.0
        self.alpha_gw= 1.0
        self.h=1e-8


        #Noise parameters
        self.sigma_p= 1e-6
        self.sigma_m= 1e-13

