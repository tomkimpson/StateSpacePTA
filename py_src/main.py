# import sys
# from run import bilby_inference_run,pp_plot_run

# #Currently these are passed as command line arguments
# #Convenient, but could be better to setup a json/config file to read from for reproducibility 
# arg_name          = sys.argv[1]           # reference name
# h                 =  float(sys.argv[2])   # strain
# measurement_model =  sys.argv[3] # whether to use the H0(null) or H1(earth/pulsar) model
# seed              = int(sys.argv[4])          # the seeding
# num_gw_sources    =  int(sys.argv[5])


# if __name__=="__main__":
#     bilby_inference_run(arg_name,h,measurement_model,seed,num_gw_sources)


from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter
import arviz as az
import matplotlib.pyplot as plt 


import pymc as pm
import numpy as np 
from pytensor_op import LogLike
import pytensor.tensor as pt
#import theano.tensor as tt
# print(f"Running on PyMC v{pm.__version__}")

import matplotlib.pyplot as plt 

if __name__=="__main__":

        
    h = 5e-15
    measurement_model = 'pulsar'
    seed = 903293933
    num_gw_sources = 1
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,num_gw_sources=num_gw_sources) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA,P)
    logl = LogLike(KF) #wrap it


    # # use PyMC to sampler from log-likelihood
    with pm.Model():
        

        omega_exponent = pm.Uniform("omega_exp", lower=-8, upper=-6)
        phi0   = pm.Uniform("phi", lower=0.00, upper=np.pi/2)
        
        psi    = pm.Uniform("psi", lower=0.00, upper=np.pi)
        iota   = pm.Uniform("iota", lower=0.00, upper=np.pi/2)
        delta  = pm.Uniform("delta", lower=0.00, upper=np.pi/2)
        alpha  = pm.Uniform("alpha", lower=0.00, upper=np.pi)

        theta = pt.as_tensor_variable([phi0,psi,iota,delta,alpha,omega_exponent])
        #theta = pt.as_tensor_variable([phi0,omega_exponent])


        pm.Potential("likelihood", logl(theta))

        #pm.sample(5000,tune=15000,target_accept=0.95)
        #https://discourse.pymc.io/t/nuts-sampler-effective-samples-is-smaller-than-200-for-some-parameters/5393


        #idata_mh = pm.sample(6000, tune=6000,discard_tuned_samples=True)
        #idata_mh = pm.sample(10000,tune=15000,discard_tuned_samples=True)
        idata_mh = pm.sample(draws=4,tune=500,discard_tuned_samples=True)


        idata_mh.to_netcdf("filename.nc")

    
    az.plot_trace(idata_mh)
    plt.show()



















# scratch space

    ## TEST WHAT TH ELIKELIHOOD CURVES LOOK LIKE

    # NN = 1000
    # likelihoods = np.zeros(NN)
    # xx = np.linspace(0,np.pi,NN)
    # phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw = 0.20,2.50,1.0,1.0,1.0
    # for i,psi in enumerate(xx):
    #     theta = np.array([phi0_gw,psi,iota_gw,delta_gw,alpha_gw])
    #     likelihoods[i] = KF.likelihood(theta)

    
    # plt.plot(xx,likelihoods)
    # plt.show()

    # ## END 


    # import sys 
    # sys.exit()