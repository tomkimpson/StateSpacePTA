




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from plotting import plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from priors import priors_dict
import numpy as np
import time 

if __name__=="__main__":


    P   = SystemParameters(Npsr=0)       # define the system parameters as a class
    PTA = Pulsars(P)                     # setup the PTA
    data = SyntheticData(PTA,P)          # generate some synthetic data

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)


    #Specify the parameters
    guessed_parameters = priors_dict(PTA,P)

    # Run the KF for the first time
    t1 = time.perf_counter()
    model_likelihood= KF.likelihood(guessed_parameters)
    t2 = time.perf_counter()
    print("Runtime 1: ", t2-t1)

    # Run the KF for the second time
    t1 = time.perf_counter()
    model_likelihood = KF.likelihood(guessed_parameters)
    t2 = time.perf_counter()
    print("Runtime 2: ", t2-t1)







