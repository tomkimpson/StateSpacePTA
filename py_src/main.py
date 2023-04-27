




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


    P   = SystemParameters(Npsr=1)       #define the system parameters as a class
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
   #print(data.phase_i)
    






    # #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # # Run the KF once with the correct parameters.
    guessed_parameters = priors_dict(P,data)

    model_likelihood,state_predictions = KF.likelihood(guessed_parameters)
    print("Inferred omega:", state_predictions[-1,1])
    print("Likelihood:", model_likelihood)



    guessed_parameters = priors_dict(P,data)
    guessed_parameters["phase_i"] = np.random.uniform(low=0.0, high=np.pi, size=len(PTA.f)) #np.randomdata.phase_i * 1.1
    guessed_parameters["Ai"] = 1e-9 #np.random.uniform(low=np.min(data.Ai), high=np.min(data.Ai), size=len(PTA.f)) #np.randomdata.phase_i * 1.1

    model_likelihood,state_predictions = KF.likelihood(guessed_parameters)
    print("Inferred omega:", state_predictions[-1,1])
    print("Likelihood:", model_likelihood)



    import matplotlib.pyplot as plt 
    plt.plot(PTA.t, state_predictions[:,1])
    plt.show()

    # phis = np.arange(0.0,1.0,0.1)


    # likelihoods = np.zeros_like(phis)
    # for i in range(len(phis)):
    #     print(i, phis[i])
    #     g = guessed_parameters.copy()
    #     g["phi0_gw"] = phis[i]


        
    #     model_likelihood,state_predictions = KF.likelihood(g)

    #     print(i, phis[i],model_likelihood)


    #     likelihoods[i] = model_likelihood
   





    # guessed_parameters["Ai"] = data.Ai*1.10
    # model_likelihood,state_predictions = KF.likelihood(guessed_parameters)
   
    # print("Inferred omega:", state_predictions[-1,1])
    # print("Likelihood:", model_likelihood)





    import matplotlib.pyplot as plt 
    # plt.plot(phis, likelihoods)
    # plt.show()

