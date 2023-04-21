




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

from plotting import plot_all

if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")


    P   = SystemParameters()       #define the system parameters as a class
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


    # Run the KFwith the correct parameters
    true_parameters = priors_dict(PTA,P)
    model_likelihood, model_state_predictions = KF.likelihood_and_states(true_parameters)
    print("Model likelihood is: ", model_likelihood)
    plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 1,savefig=None)

   
    #Bilby 
    #init_parameters, priors = bilby_priors_dict(PTA,P)
    #BilbySampler(KF,init_parameters,priors,label="example_2_parameters",outdir="../data/nested_sampling/")







