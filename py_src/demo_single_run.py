




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



if __name__=="__main__":


    P    = SystemParameters(Npsr=0)       #define the system parameters as a class
    PTA  = Pulsars(P)               #setup the PTA
    data = SyntheticData(PTA,P) #generate some synthetic data
   

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # Run the KFwith the correct parameters
    guessed_parameters = priors_dict(PTA,P)
    model_likelihood, model_state_predictions = KF.likelihood_and_states(guessed_parameters)
    print("likelihood = ", model_likelihood)
    plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 0)







