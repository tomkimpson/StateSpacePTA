

import numpy as np 

"""
Function that returns a dict of parameters which define the system
"""
def SystemParameters(NF=np.float64,    # the number format of the arguments
                     T = 10,           # how long to integrate for in years
                     cadence=7,        # the interval between observations
                     Ω=   5.0e-7,        # GW angular frequency
                     Φ0 = 0.20,        # GW phase offset at t=0
                     ψ =  2.50,        # GW polarisation angle
                     ι = 1.0,          # GW source inclination
                     δ =  1.0,         # GW source declination
                     α =  1.0,         # GW source right ascension
                     h =  1e-2,        # GW plus strain
                     σp = 1e-13,       # process noise standard deviation
                     σm = 1e-8,        # measurement noise standard deviation
                     Npsr = 0,         # Number of pulsars to use in PTA. 0 = all
                     use_psr_terms_in_data=True, # when generating the synthetic data, include pulsar terms?
                     use_psr_terms_in_model=True # do you want the pulsar terms to be in the Kalman measurement model?
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
               "Npsr":     Npsr,
               "psr_terms_data":use_psr_terms_in_data,
               "psr_terms_model":use_psr_terms_in_model})

    return data
   



disable_JIT = False  
