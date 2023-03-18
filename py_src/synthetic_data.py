

import sdeint
import numpy as np 

from gravitational_waves import gw_prefactor,gw_modulation
class SyntheticData:
    
    
    
    
    def __init__(self,pulsars,GW,seed):


        t = pulsars.t
        Npsr = len(pulsars.f)

        f0 = pulsars.f#.reshape(Npsr,1)
        fdot = pulsars.fdot#.reshape(Npsr,1)
        gamma = pulsars.gamma#.reshape(Npsr,1)
        sigma_p = np.full((Npsr,1),pulsars.sigma_p**2)
        

        def f(x,t): 
            return -gamma * x + gamma*(f0 + fdot*t) + fdot  
        def g(x,t): 
            return sigma_p

       
        self.intrinsic_frequency = sdeint.itoint(f,g,f0, t)

        
        prefactor, dot_product =gw_prefactor(GW.n,pulsars.q, GW.Hij, GW.omega_gw, pulsars.d)




        f_measured_clean = np.zeros((len(t),Npsr))


        for i in range(len(t)):

           GW_factor = gw_modulation(t[i], GW.omega_gw,GW.phi0_gw,prefactor,dot_product)
           f_measured_clean[i,:] = self.intrinsic_frequency[i,:] * GW_factor
    

        measurement_noise = np.random.normal(0, pulsars.sigma_m,f_measured_clean.shape) # Measurement noise
 
        self.f_measured = f_measured_clean + measurement_noise

        


    # f_measured = add_gauss(f_measured_clean,σm,0.0)
    # # f_measured = zeros(NF,size(q)[1],length(t))
    # # for i=1:size(q)[1]
    # #   println
    # #   f_measured[i,:] = add_gauss(f_measured_clean[i,:], σm, 0.0) #does this do the correct thing?   
    # # end
    
    # return intrinsic_frequency,f_measured