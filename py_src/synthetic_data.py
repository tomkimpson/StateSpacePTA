

import sdeint
import numpy as np 

from gravitational_waves import gw_prefactor_optimised,gw_prefactor_optimised_exp,gw_prefactor_optimised_trig
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
        modulation_factors = gw_prefactor_optimised(
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
        
        # #Now calculate the modulation factor due to the GW
        # modulation_factors1 = gw_prefactor_optimised_exp(
        #                        P["delta_gw"],
        #                        P["alpha_gw"],
        #                        P["psi_gw"],
        #                        pulsars.q,
        #                        pulsars.q_products,
        #                        P["h"],
        #                        P["iota_gw"],
        #                        P["omega_gw"],
        #                        pulsars.d,
        #                        pulsars.t,
        #                        P["phi0_gw"]
        #                        )
        

        #         #Now calculate the modulation factor due to the GW
        # modulation_factors2 = gw_prefactor_optimised_trig(
        #                        P["delta_gw"],
        #                        P["alpha_gw"],
        #                        P["psi_gw"],
        #                        pulsars.q,
        #                        pulsars.q_products,
        #                        P["h"],
        #                        P["iota_gw"],
        #                        P["omega_gw"],
        #                        pulsars.d,
        #                        pulsars.t,
        #                        P["phi0_gw"]
        #                        )



        # print("compare mods")
        # print(modulation_factors1 == modulation_factors2)
        # print(modulation_factors1 - modulation_factors2)





        #The measured frequency, no noise
        f_measured_clean= self.intrinsic_frequency * modulation_factors

        #...and now add some mean zero Gaussian noise
        measurement_noise = np.random.normal(0, pulsars.sigma_m,f_measured_clean.shape) # Measurement noise
        self.f_measured = f_measured_clean + measurement_noise

