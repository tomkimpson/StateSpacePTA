

import numpy as np 


"""
Function that returns a dict of parameters which define the system
"""

def SystemParameters(NF=np.float64,    # the number format of the arguments
                     T = 10,           # how long to integrate for in years
                     cadence=7,        # the interval between observations
                     Ω=   5e-7,           # GW angular frequency
                     Φ0 = 0.20,        # GW phase offset at t=0
                     ψ =  2.50,         # GW polarisation angle
                     ι =  1.0,          # GW source inclination
                     δ =  1.0,          # GW source declination
                     α =  1.0,          # GW source right ascension
                     h =  1e-2,         # GW strain
                     σp = 1e-13,       # process noise standard deviation
                     σm = 1e-10,        # measurement noise standard deviation
                     Npsr = 0          # Number of pulsars to use in PTA. 0 = all
                     ): 

    data = dict({
               "NF":       NF, 
               "T":        NF(T),
               "cadence":  NF(cadence),
               "omega_gw": NF(Ω),
               "phi0_gw":  NF(Φ0),
               "psi_gw":   NF(ψ),
               "iota_gw":  NF(ι),
               "delta_gw": NF(δ),
               "alpha_gw": NF(α),
               "h":        NF(h),
               "sigma_p":  NF(σp),
               "sigma_m":  NF(σm),
               "Npsr":     Npsr})

    return data
   


class SystemParams:


    def __init__(self,NF=np.float64,    # the number format of the arguments
                     T = 10,           # how long to integrate for in years
                     cadence=7,        # the interval between observations
                     Ω=   5e-7,           # GW angular frequency
                     Φ0 = 0.20,        # GW phase offset at t=0
                     ψ =  2.50,         # GW polarisation angle
                     ι =  1.0,          # GW source inclination
                     δ =  1.0,          # GW source declination
                     α =  1.0,          # GW source right ascension
                     h =  1e-2,         # GW strain
                     σp = 1e-13,       # process noise standard deviation
                     σm = 1e-10,        # measurement noise standard deviation
                     Npsr = 0          # Number of pulsars to use in PTA. 0 = all
                     ):
        
        self.NF = NF 
        self.T = T 
        self.cadence=cadence
        self.omega_gw = Ω
        self.phi0_gw = Φ0
        self.psi_gw = ψ
        self.iota_gw = ι 
        self.delta_gw = δ 
        self.alpha_gw = α  
        self.h = h 
        self.sigma_p = σp
        self.sigma_m =σm
        self.Npsr = Npsr

        