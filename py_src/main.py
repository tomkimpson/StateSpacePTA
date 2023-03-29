




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from plotting import plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict
from bilby_wrapper import BilbyLikelihood

import numpy as np
import bilby


if __name__=="__main__":


    P   = SystemParameters(Npsr=3)       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    #GW  = GWs(P)                   #setup GW related constants and functions. This is a dict, not a class, for interaction later with Bilby 
    data = SyntheticData(PTA,P) #generate some synthetic data
    # plot_statespace(PTA.t,data.intrinsic_frequency,data.f_measured,1) #plot it if needed

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # Run the KF once with the correct parameters
    
    guessed_parameters = priors_dict(PTA,P)
    #guessed_parameters["omega_gw"] = 1e-3
    model_likelihood, model_state_predictions, model_covariance_predictions = KF.likelihood(guessed_parameters)
    print("likelihood = ", model_likelihood)
    # # t,states,measurements,predictions,psr_index
    plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 0)

    #Bilby 
    #init_parameters, priors = bilby_priors_dict(PTA)
    #BilbySampler(KF,init_parameters,priors)







