




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

    # # # Run the KF once with the correct parameters
    # guessed_parameters = priors_dict(PTA,GW)
    # # print(guessed_parameters)
    # model_likelihood, model_state_predictions, model_covariance_predictions = KF.likelihood(guessed_parameters,"H0")
    # # print("likelihood = ", model_likelihood)
    # # # t,states,measurements,predictions,psr_index
    # plot_all(PTA.t, data.intrinsic_frequency, data.f_measured, model_state_predictions, 0)

    #Bilby 
    init_parameters, priors = bilby_priors_dict(PTA)


    #Manually specify the injection parameters
    #todo: automate this for generality
    injection_parameters = dict(
        omega_gw=5e-7,
        phi0_gw=0.20,
        psi_gw=2.50,
        iota_gw=1.0,
        delta_gw=1.0,
        alpha_gw=1.0,
        h=1e-2,
        f0=327.8470205611185,
        fdot=-1.227834e-15,
        distance=181816860005.41092,
        gamma=1e-13,
        sigma_p=1e-8,
        sigma_m=1e-08
    )






    print(init_parameters)
    print(priors)
    BilbySampler(KF,init_parameters,priors,injection_parameters,"PTA1", "../data/nested_sampling")




























# SCRATCH SPACE


    #use the below to generate a rough plot of likelihood vs parameter








#     xx = []
#     yy = []
#     zz = []
#     N = 100
#     omegas = np.logspace(-8,-6,num=N)
#     omegas = np.linspace(9e-8,1.1e-7,num=N)
#     omegas=np.arange(9e-8,1.1e-7,1e-9)
#     for i in range(len(omegas)):
#         #guessed_parameters = priors.sample()
#         guessed_parameters = priors_dict(PTA,GW)

#         #omega = guessed_parameters["omega_gw"]
#         guessed_parameters["omega_gw"] = omegas[i]
#         model_likelihood,model_state_predictions,P = KF.likelihood(guessed_parameters)

#         xx.extend([guessed_parameters["omega_gw"]])
#         yy.extend([abs(model_likelihood)])
#         zz.extend([P[0]])


# #    # xx.extend([1e-7])

# #     guessed_parameters = priors_dict(PTA,GW)
# #     guessed_parameters["omega_gw"] = 1e-6
# #     model_likelihood,model_state_predictions,P = KF.likelihood(guessed_parameters)
# #    # print(P)
# #     yy.extend([abs(model_likelihood)])
# #     .extend([abs(model_likelihood)])

# #     print(model_likelihood)

#     import matplotlib.pyplot as plt
#     plt.scatter(xx,zz)
#     plt.plot(xx,zz)

#     plt.xscale('log')
#     plt.axvline(1e-7,linestyle="--", c = '0.5')
#     plt.yscale('log')
#     plt.xlabel("omega")
#     plt.ylabel("log L ")

#     plt.show()



    

