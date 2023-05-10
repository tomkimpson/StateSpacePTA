




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

arg_name = sys.argv[1]  #reference name
h        =  float(sys.argv[2]) #strain
measurement_model =  sys.argv[3] #whether to use the H0 or H1 model

from numba import jit, config

from system_parameters import disable_JIT
config.DISABLE_JIT = disable_JIT

if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")


    P   = SystemParameters(h=h,σp=0.0,σm=1e-13,use_psr_terms_in_data=True,measurement_model=measurement_model)       # define the system parameters as a class
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
    print("True strain = ", h)
    #from plotting import plot_all
    #plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, state_predictions,measurement_predictions, 1,savefig=None)

   
    #Bilby 
    init_parameters, priors = bilby_priors_dict(PTA,P)
    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/")







