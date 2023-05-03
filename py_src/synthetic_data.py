

import sdeint
import numpy as np 

from gravitational_waves import gw_synthetic_data
class SyntheticData:
    
    
    
    
    def __init__(self,pulsars,P):


        #Load some PTA related quantities
        t = pulsars.t
        Npsr = pulsars.Npsr 

        #...and the pulsar parameters
        f0 = pulsars.f
        fdot = pulsars.fdot
        gamma = pulsars.gamma
        sigma_p = np.full((Npsr,1),pulsars.sigma_p)
        

        #First get the intrinstic pulsar evolution by solving the ito equation
        def f(x,t): 
            return -gamma * x + gamma*(f0 + fdot*t) + fdot  
        def g(x,t): 
            return sigma_p

       
        self.intrinsic_frequency = sdeint.itoint(f,g,f0, t)

        #Now calculate the modulation factor due to the GW
        modulation_factors = gw_synthetic_data(
                               P["delta_gw"],
                               P["alpha_gw"],
                               P["psi_gw"],
                               pulsars.q,
                               pulsars.q_products,
                               P["h"],
                               P["iota_gw"],
                               P["omega_gw"],
                               pulsars.d,
                               pulsars.t,
                               P["phi0_gw"]
                               )
        



        # print("DTYPES")
        # print(f0.dtype)
        # print(fdot.dtype)
        # print(gamma.dtype)
        # print(sigma_p.dtype)


       # print(self.intrinsic_frequency.dtype)
        #print(modulation_factors.dtype)
     

        #The measured frequency, no noise
        self.f_measured_clean= self.intrinsic_frequency * modulation_factors

        #print(self.f_measured_clean.dtype)

        #...and now add some mean zero Gaussian noise


       # rng = np.random.default_rng()
       # measurement_noise = rng.standard_normal(self.f_measured_clean.shape,dtype=P["NF"]) * pulsars.sigma_m



        measurement_noise = np.random.normal(0, pulsars.sigma_m,self.f_measured_clean.shape) # Measurement noise



        #print(measurement_noise.dtype)


        #print("The measurement noise is:", measurement_noise)
        self.f_measured = self.f_measured_clean + measurement_noise

