


from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter

from priors import optimal_parameter_values,sub_optimal_parameter_values,log_probability
import logging 
import emcee
import numpy as np 


def emcee_inference_run(arg_name,h,measurement_model,seed):

    logger = logging.getLogger().setLevel(logging.INFO)
    #Setup the system
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    optimal_parameters = optimal_parameter_values(PTA,P)
    model_likelihood = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")
    


    logging.info(f"Starting emcee inference attempt")
    starting_parameter_guess = sub_optimal_parameter_values(PTA,P)
    pos = starting_parameter_guess + 1e-4 * np.random.randn(32, len(starting_parameter_guess))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(KF)
    )
    sampler.run_mcmc(pos, 5000, progress=True)

    # #Bilby
    # init_parameters, priors = bilby_priors_dict(PTA,P)
   

    # logging.info("Testing KF using parameters sampled from prior")
    # params = priors.sample(1)
    # model_likelihood = KF.likelihood(params)
    # logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    
    # # #Now run the Bilby sampler
    # BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/")
    # logging.info("The run has completed OK")

