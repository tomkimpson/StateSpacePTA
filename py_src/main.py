




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
#from plotting import plot_all
from model import EKF
from kalman_filter import KalmanFilter
#from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict
#from bilby_wrapper import BilbyLikelihood

import numpy as np
#import bilby
import sys



from plotting import plot_measurement


arg_name = sys.argv[1]

from numba import jit, config

from system_parameters import disable_JIT
config.DISABLE_JIT = disable_JIT

if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")


    P   = SystemParameters(Npsr=3)       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    data = SyntheticData(PTA,P) #generate some synthetic data
    
    GW_period = 2*np.pi/(P["omega_gw"])
    approx_num_cycles = PTA.t[-1] / GW_period
    print("Approx number of GW cycles:", approx_num_cycles)

    #plot_measurement(PTA.t,data.f_measured,2)


    #Define the model 
    model = EKF

    # print(PTA.f)
    # print(data.Ai)
    # print(data.phase_i)
    


    # #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA,data.Ai, data.phase_i % 2*np.pi)

    # # Run the KF once with the correct parameters.
    #guessed_parameters = priors_dict(PTA,P)
    model_likelihood,state_predictions = KF.likelihood()
   


    import matplotlib.pyplot as plt 
    plt.plot(PTA.t, state_predictions[:,1])
    plt.show()

