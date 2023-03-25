

import numpy as np 


"""
Function that returns a dict of parameters which define the system
"""
def SystemParameters(): #this is a dict, not a class

    
    data = dict({"NF": np.longdouble,
               "T": 10,
               "cadence": 7.0,

               "omega_gw": 5e-7,
               "phi0_gw":0.20,
               "psi_gw":2.50,
               "iota_gw": 0.0,
               "delta_gw":0.0,
               "alpha_gw":1.0,
               "h": 1e-2,
               "sigma_p": 1e-8,
               "sigma_m":1e-10})

    return data
   

