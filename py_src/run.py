


from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import bilby_priors_dict
import logging 
import numpy as np 

def bilby_inference_run(arg_name,h,measurement_model,seed,num_gw_sources):

    logger = logging.getLogger().setLevel(logging.INFO)
    #Setup the system
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,num_gw_sources=num_gw_sources) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #This allows JIT precompile

    init_parameters_optimal, priors_optimal = bilby_priors_dict(PTA,P,set_parameters_as_known=True)
    optimal_parameters = priors_optimal.sample(1)
    model_likelihood = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")
    logging.info("The optimal parameters used to generate the data are as follows:")
    
    for key,value in optimal_parameters.items():
        print(key, value)
    
    #Bilby
    init_parameters, priors = bilby_priors_dict(PTA,P)
    logging.info("Testing KF using parameters sampled from prior")
    params = priors.sample(1)
    model_likelihood = KF.likelihood(params)
    logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    

    #Now run the Bilby sampler
    #How many parameters are we trying to infer?
    num_params = len(params) - P.Npsr #subtract the N \gammas which we never infer, just carry
    logging.info(f"Total number of parameters to be inferred = {len(params)}")
    #...set the number of live points accordingly
    npoints = len(params) * 3
    if npoints < 1000:
        npoints=1000

    logging.info(f"Number of live points is {npoints}")

    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/",npoints=npoints)
    logging.info("The run has completed OK")






def pp_plot_run(arg_name,h,measurement_model,seed,omega,phi_0,psi,alpha,delta):

    logger = logging.getLogger().setLevel(logging.INFO)
    #Setup the system
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,Ω=omega,Φ0=phi_0,ψ=psi,α=alpha,δ=delta) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    optimal_parameters = priors_dict(PTA,P)
    model_likelihood = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")
    
    #Bilby
    init_parameters, priors = bilby_priors_dict(PTA,P)
   

    logging.info("Testing KF using parameters sampled from prior")
    params = priors.sample(1)


    print("The sampled params are as follows")
    print(params)


    model_likelihood = KF.likelihood(params)
    logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    
    # #Now run the Bilby sampler
    npoints=2000
    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/",npoints=npoints)
    logging.info("The run has completed OK")

