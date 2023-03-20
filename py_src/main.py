




from system_parameters import SystemParameters
from pulsars import Pulsars
from gravitational_waves import GWs
from synthetic_data import SyntheticData
from plotting import plot_statespace,plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict

import numpy as np
import bilby














if __name__=="__main__":


    P   = SystemParameters()       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    GW  = GWs(P)                   #setup GW related constants and functions. This is a dict, not a class, for interaction later with Bilby 
    data = SyntheticData(PTA,GW,1) #generate some synthetic data
    #plot_statespace(PTA.t,data.intrinsic_frequency,data.f_measured,1)

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # # #Run the KF
    # guessed_parameters = priors_dict(PTA,GW)
    
    # model_state_predictions,model_likelihood = KF.likelihood(guessed_parameters)
    # print(model_likelihood)
    # plot_all(PTA.t,data.intrinsic_frequency,data.f_measured,model_state_predictions,1)

    #Bilby 
    
    init_parameters,priors = bilby_priors_dict(PTA)
    #guessed_parameters = priors.sample()
    guessed_parameters = priors_dict(PTA,GW)

    print(guessed_parameters["omega_gw"])
    model_likelihood,model_state_predictions = KF.likelihood(guessed_parameters)
    plot_all(PTA.t,data.intrinsic_frequency,data.f_measured,model_state_predictions,1)

    #BilbySampler(KF,init_parameters,priors)
    

