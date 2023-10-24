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

print(f"Running on PyMC v{pm.__version__}")

import matplotlib.pyplot as plt 

if __name__=="__main__":

        
    h = 5e-15
    measurement_model = 'pulsar'
    #seed = 1237
    seed = 903293933
    num_gw_sources = 1
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,num_gw_sources=num_gw_sources) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA,P)


    ## TEST WHAT TH ELIKELIHOOD CURVES LOOK LIKE
    omega = 5e-7 
    NN = 1000
    likelihoods = np.zeros(NN)
    xx = np.linspace(0,np.pi/2,NN)
    for i,phi in enumerate(xx):
        theta = np.array([omega,phi])
        likelihoods[i] = KF.likelihood(theta)

    
    plt.plot(xx,likelihoods)
    plt.show()

    ## END 


    import sys 
    sys.exit()

    logl = LogLike(KF) #wrap it


    # # use PyMC to sampler from log-likelihood
    with pm.Model():
        # uniform priors on m and c
        #omega = pm.Uniform("m", lower=4e-7, upper=6e-7)
        omega = pm.Uniform("m", lower=1e-8, upper=1e-5)

        phi0 = pm.Uniform("c", lower=0.00, upper=np.pi/2)
    

    #     # convert m and c to a tensor vector
        theta = pt.as_tensor_variable([omega,phi0])

    #     # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))

    #     # Use custom number of draws to replace the HMC based defaults
        #idata_mh = pm.sample(3000, tune=1000)
        idata_mh = pm.sample(1000, tune=1000,discard_tuned_samples=True)
        idata_mh.to_netcdf("filename.nc")

    
    az.plot_trace(idata_mh, lines=[("m", {}, 5e-7),("c", {}, 0.20)])
    plt.show()

