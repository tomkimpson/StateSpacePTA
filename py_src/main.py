




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

arg_name = sys.argv[1]

from numba import jit, config

from system_parameters import disable_JIT
config.DISABLE_JIT = disable_JIT

if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")


    P   = SystemParameters(h=1e-10,σm=1e-12,σp=1e-15,Npsr=20)       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    data = SyntheticData(PTA,P) #generate some synthetic data
    

    #Define the model 
    model = LinearModel


    #Scale the data
    #print(data.f_measured.shape)
    #print(PTA.f.shape)


    #scaled_measurement_data = data.f_measured / PTA.f
    #scaled_measurement_data = data.f_measured / PTA.f


    



    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)


    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    guessed_parameters = priors_dict(PTA,P)
    model_likelihood = KF.likelihood(guessed_parameters)
    print("Ideal likelihood = ", model_likelihood)
   
    # from plotting import plot_all
    # plot_all(PTA.t, data.intrinsic_frequency / PTA.f, scaled_measurement_data, model_state_predictions, 1,savefig=None)
    # wrong_parameters = erroneous_priors_dict(PTA,P,1e-1)
    # model_likelihood = KF.likelihood(wrong_parameters)
    # print("Wrong likelihood = ", model_likelihood)


    #Bilby 
    init_parameters, priors = bilby_priors_dict(PTA,P)
    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/")







