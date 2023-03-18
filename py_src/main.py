




from system_parameters import SystemParameters
from pulsars import Pulsars
from gravitational_waves import GWs
from synthetic_data import SyntheticData
from plotting import plot_simple
from model import LinearModel
from kalman_filter import KalmanFilter
from priors import Priors 

if __name__=="__main__":


    P   = SystemParameters()       #define the system parameters
    PTA = Pulsars(P)               #setup the PTA
    GW  = GWs(P)                   #setup GW related constants and functions 
    data = SyntheticData(PTA,GW,1) #generate some synthetic data
    #plot_simple(PTA.t,data.intrinsic_frequency,data.f_measured,1)


    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA.dt)


    guessed_parameters = Priors(PTA,GW)
    KF.likelihood(guessed_parameters)