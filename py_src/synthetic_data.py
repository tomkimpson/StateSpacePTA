

import sdeint
import numpy as np 

from gravitational_waves import gw_earth_terms,gw_psr_terms

#from mpmath import mp, mpf 
#mp.dps = 50
class SyntheticData:
    
    
    
    
    def __init__(self,pulsars,P):


        #Load some PTA related quantities
        t = pulsars.t
        Npsr = pulsars.Npsr 

        #...and the pulsar parameters
        f0 = pulsars.f#* mpf(1.0)
        fdot = pulsars.fdot#* mpf(1.0)
        gamma = pulsars.gamma #* mpf(1.0)
        sigma_p = np.full((Npsr,1),pulsars.sigma_p)#* mpf(1.0)
        
        #print("dt = ", t[1] - t[0])
        #print("gamma = ", gamma)

        #First get the intrinstic pulsar evolution by solving the ito equation
        def f(x,t):
            #print(-gamma * x + gamma*(f0 + fdot*t) + fdot)
            return -gamma * x + gamma*(f0 + fdot*t) + fdot  
        def g(x,t): 
            return sigma_p

        generator = np.random.default_rng(1234)
        self.intrinsic_frequency = sdeint.itoint(f,g,f0, t,generator=generator)

        #Now calculate the modulation factor due to the GW
        if P["psr_terms_data"]:
            mod_factor = gw_psr_terms
            print("Attention: You are including the PSR terms in your synthetic data generation")
        else:
            mod_factor = gw_earth_terms
            print("Attention: You are using just the Earth terms in your synthetic data generation")


        modulation_factors = mod_factor(
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
        
        #The measured frequency, no noise
        self.f_measured_clean= self.intrinsic_frequency * modulation_factors
        measurement_noise = generator.normal(0, pulsars.sigma_m,self.f_measured_clean.shape) # Measurement noise. Seeded
        self.f_measured = self.f_measured_clean + measurement_noise

        if P["heterodyne"]:
            print("Heterodyning the measured data relative to a reference ephemeris")
            self.f_measured = P["heterodyne_scale_factor"]*(self.f_measured - pulsars.ephemeris)
        else:
            print("No heterodyne corrections")
