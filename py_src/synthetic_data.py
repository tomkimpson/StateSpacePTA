

import sdeint
import numpy as np 

from gravitational_waves import compute_prefactors,gw_modulation



import sys
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



        self.Ai, self.phase_i = compute_prefactors(P["omega_gw"],P["delta_gw"],P["alpha_gw"],P["psi_gw"],P["h"], P["iota_gw"],pulsars.q,pulsars.q_products,pulsars.d)

      
    
        modulation_factors = np.zeros((len(t),Npsr))
        for i in range(len(t)):
            gw_phase = P["omega_gw"]*t[i] + P["phi0_gw"]
            modulation_factors[i,:] = gw_modulation(self.Ai,gw_phase,self.phase_i)

       

        #The measured frequency, no noise
        f_measured_clean= self.intrinsic_frequency * modulation_factors

        #...and now add some mean zero Gaussian noise
        measurement_noise = np.random.normal(0, pulsars.sigma_m,f_measured_clean.shape) # Measurement noise
        self.f_measured = f_measured_clean + measurement_noise

