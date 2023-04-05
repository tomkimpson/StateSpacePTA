




from system_parameters import SystemParameters
from pulsars import Pulsars
#from gravitational_waves import GWs
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


   

    P   = SystemParameters(Npsr=0)       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    #GW  = GWs(P)                   #setup GW related constants and functions. This is a dict, not a class, for interaction later with Bilby 
    data = SyntheticData(PTA,P) #generate some synthetic data


    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    print("Running once")
    true_parameters = priors_dict(PTA,P)
    t1 = time.perf_counter()
    model_likelihood = KF.likelihood(true_parameters)
    t2 = time.perf_counter()
    print("first time:", t2-t1)

    #print("The log likelihood with the true parameters is: ", model_likelihood)


    # print("Running again")
    # true_parameters = priors_dict(PTA,P)
    # t3 = time.perf_counter()
    # model_likelihood,model_state_predictions = KF.likelihood_and_states(true_parameters)
    # t4 = time.perf_counter()
    # print("second time:", t4-t3)

    # plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 0)

    #plot_statespace(PTA.t,data.intrinsic_frequency,data.f_measured,1) #plot it if needed


    # #Bilby 
    init_parameters, priors = bilby_priors_dict(PTA,P)
    dlogz=0.10
    BilbySampler(KF,init_parameters,priors,"likelihood_test51","../data/nested_sampling",dlogz)

   


    #BilbySampler(KF,init_parameters,priors,"likelihood_test49","../data/nested_sampling",dlogz)


















############# SCRATCH SPACE

    # plot_statespace(PTA.t,data.intrinsic_frequency,data.f_measured,1) #plot it if needed

































 # # Run the KF once with the correct parameters
    #guessed_parameters = priors_dict(PTA,GW)
    # # print(guessed_parameters)
    # t1 = time.perf_counter()
    # #model_likelihood, model_state_predictions, model_covariance_predictions = KF.likelihood(guessed_parameters)
    # model_likelihood = KF.likelihood(guessed_parameters)

    # t2=time.perf_counter()
    # print("Runtime = ", t2-t1)
    # print("likelihood = ", model_likelihood)
    # # t,states,measurements,predictions,psr_index
    #plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 0)












    
   #p = priors.sample()
   # 
    #print(p)
   # model_likelihood = KF.likelihood(p)
   # print(p["f00"], p["omega_gw"])
    #print(model_likelihood)


    # for i in range(100):
    #     p = priors.sample()
    
    # #t1 = time.perf_counter()
    # #model_likelihood, model_state_predictions, model_covariance_predictions = KF.likelihood(guessed_parameters)
    #     model_likelihood = KF.likelihood(p)

    #     #t2=time.perf_counter()
    # #print("Runtime = ", t2-t1)
    #     print("likelihood = ", i,  model_likelihood)




    #guessed_parameters = priors_dict(PTA,GW)
    #print(guessed_parameters)

#327.8470205611185

    #fs = np.arange(325,330,0.10)
    # fs = np.arange(1e-3,1e-1,1e-3)

    # xx = []
    # yy = []
    # for i in fs:
    #     guessed_parameters = priors_dict(PTA,GW)
        
    #     #guessed_parameters["f00"] = i 
    #     guessed_parameters["h"] = i 

        
    #     guessed_parameters["sigma_p"] = 1e-3
    #     guessed_parameters["omega_gw"] = 1e-7
        
        
        
    #     model_likelihood = KF.likelihood(guessed_parameters)

    #     xx.extend([i])
    #     yy.extend([model_likelihood])



    # print(xx)
    # print(yy)    
    # import matplotlib.pyplot as plt 
    # plt.scatter(xx,yy)
    # plt.axvline(1e-2)
    # plt.show()