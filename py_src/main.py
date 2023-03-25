




from system_parameters import SystemParameters
from pulsars import Pulsars
from gravitational_waves import GWs
from synthetic_data import SyntheticData
from plotting import plot_statespace,plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict
from bilby_wrapper import BilbyLikelihood

import numpy as np
import bilby

import time 

if __name__=="__main__":


    P   = SystemParameters()       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    GW  = GWs(P)                   #setup GW related constants and functions. This is a dict, not a class, for interaction later with Bilby 
    data = SyntheticData(PTA,GW,1) #generate some synthetic data
    # plot_statespace(PTA.t,data.intrinsic_frequency,data.f_measured,1) #plot it if needed

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # # Run the KF once with the correct parameters
    # guessed_parameters = priors_dict(PTA,GW)
    # # print(guessed_parameters)
    # t1 = time.perf_counter()
    # #model_likelihood, model_state_predictions, model_covariance_predictions = KF.likelihood(guessed_parameters)
    # model_likelihood = KF.likelihood(guessed_parameters)

    # t2=time.perf_counter()
    # print("Runtime = ", t2-t1)
    # print("likelihood = ", model_likelihood)
    # # t,states,measurements,predictions,psr_index
    #plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 0)

    # #Bilby 
    #init_parameters, priors = bilby_priors_dict(PTA)
    #BilbySampler(KF,init_parameters,priors)


    # for i in range(100):
    #     p = priors.sample()
    
    # #t1 = time.perf_counter()
    # #model_likelihood, model_state_predictions, model_covariance_predictions = KF.likelihood(guessed_parameters)
    #     model_likelihood = KF.likelihood(p)

    #     #t2=time.perf_counter()
    # #print("Runtime = ", t2-t1)
    #     print("likelihood = ", i,  model_likelihood)



    #Uncomment the below use the below to generate a rough plot of likelihood vs parameter


    xx = []
    yy = []

    N = 100

    omegas=np.arange(4e-7,6e-7,1e-9)
    hs = np.logspace(-3,-1,100)
    #for i in range(len(omegas)):
    for i in range(len(hs)):

        print(i, hs[i])
        guessed_parameters = priors_dict(PTA,GW)

        #omega = guessed_parameters["omega_gw"]
        #guessed_parameters["omega_gw"] = omegas[i]
        guessed_parameters["h"] = hs[i]
        #model_likelihood,model_state_predictions,P = KF.likelihood(guessed_parameters)
        model_likelihood = KF.likelihood(guessed_parameters)


        #xx.extend([guessed_parameters["omega_gw"]])
        xx.extend([guessed_parameters["h"]])

        yy.extend([abs(model_likelihood)])


    import matplotlib.pyplot as plt
    plt.scatter(xx,yy)
    plt.plot(xx,yy)

    plt.xscale('log')
    #plt.axvline(5e-7,linestyle="--", c = '0.5')
    plt.yscale('log')
    plt.xlabel("omega")
    plt.ylabel("log L ")

    plt.show()



    

