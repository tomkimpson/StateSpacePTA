




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
#from plotting import plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict
from bilby_wrapper import BilbyLikelihood

import numpy as np
import bilby

from dynesty import DynamicNestedSampler

from multiprocessing import Pool, freeze_support


if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")

    #pool = Pool(4)


    P   = SystemParameters(Npsr=5)       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    data = SyntheticData(PTA,P) #generate some synthetic data
    

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # Run the KF once with the correct parameters.
    # This allows JIT precompile
    guessed_parameters = priors_dict(PTA,P)
    model_likelihood = KF.likelihood(guessed_parameters)
    print("Ideal likelihood = ", model_likelihood)
   
    # #Bilby 
    # init_parameters, priors = bilby_priors_dict(PTA,P)
    # BilbySampler(KF,init_parameters,priors,label="example_2_parameters",outdir="../data/nested_sampling/")



    def prior_transform(u):


        x = np.array(u)  # copy u


        # a log-uniform distribution for omega?
        x[0] = 10**(-8 + 2 * u[0])


        # # a uniform distribution for phi0
        # transformed_parameters[1] = 0.0 + np.pi * quantile_cube[1]

        # # a uniform distribution for psi
        # transformed_parameters[2] = 0.0 + np.pi * quantile_cube[2]


        # # a uniform distribution for iota
        # transformed_parameters[3] = 0.0 + np.pi * quantile_cube[3]

        # # a uniform distribution for delta
        # transformed_parameters[4] = 0.0 + np.pi * quantile_cube[4]

        # # a uniform distribution for alpha
        # transformed_parameters[5] = 0.0 + np.pi * quantile_cube[5]

        # # a log-uniform distribution for omega
        # transformed_parameters[6] = 10**(-4 + 4 * quantile_cube[6])



        return x


    def my_likelihood(params):

        p_copy = guessed_parameters.copy()


        #omega_var,phi_var,psi_var,iota_var,delta_var,alpha_var,h_var = params 
        omega_var = params 

        
        p_copy["omega_gw"] = omega_var
        # p_copy["phi0_gw"] = phi_var
        # p_copy["psi_gw"] = psi_var
        # p_copy["iota_gw"] = iota_var
        # p_copy["delta_gw"] = delta_var
        # p_copy["alpha_gw"] = alpha_var
        # p_copy["h"] = h_var


        ll = KF.likelihood(p_copy)
        
        #print(omega_var, ll)
        return ll







    dsampler = DynamicNestedSampler(my_likelihood, prior_transform, ndim=1, bound='single') #pool=pool

    dsampler.run_nested(dlogz_init=0.05, nlive_init=500, nlive_batch=100)




