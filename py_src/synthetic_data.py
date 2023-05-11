

import sdeint
import numpy as np 

from gravitational_waves import gw_earth_terms,gw_psr_terms

from system_parameters import heterodyne, heterodyne_scale_factor
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

        np.random.seed(1234)
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
        measurement_noise = np.random.normal(0, pulsars.sigma_m,self.f_measured_clean.shape) # Measurement noise
        self.f_measured = self.f_measured_clean + measurement_noise


        
        print ("Heterodyning:", heterodyne)
        if heterodyne:
            print("Heterodyning the measured data relative to a reference ephemeris")
            self.f_measured = heterodyne_scale_factor*(self.f_measured - pulsars.ephemeris)
