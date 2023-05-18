




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
#from plotting import plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict,erroneous_priors_dict
from bilby_wrapper import BilbyLikelihood

import numpy as np
import bilby
import sys

from scipy import optimize




arg_name = sys.argv[1]

from numba import jit, config

if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")


    P   = SystemParameters(h=1e-2,σp=0.0,σm=1e-13,use_psr_terms_in_data=True,use_psr_terms_in_model=True)       # define the system parameters as a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data
    

    #Define the model 
    model = LinearModel(P)


    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)


    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    guessed_parameters = priors_dict(PTA,P)
    model_likelihood, state_predictions,measurement_predictions = KF.likelihood(guessed_parameters)
    print("Ideal likelihood = ", model_likelihood)
   
    #Scipy optimise 



    def log_likelihood(p):


        parameters_dict = guessed_parameters.copy()
        parameters_dict["omega_gw"] = p[0]

        ll,xres,yres = KF.likelihood(parameters_dict)

        return ll 




    bounds = [(1e-9, 1e-5)]


    results = optimize.shgo(log_likelihood, bounds,iters = 15)
    print(results["x"])
    # res = minimize(log_likelihood, x0, method='nelder-mead',
    #            options={'xatol': 1e-8, 'disp': True})




