


from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import bilby_priors_dict
import logging

import time
import sys 
def bilby_inference_run(arg_name,measurement_model,seed,num_gw_sources):

    logger = logging.getLogger().setLevel(logging.INFO)
    #Setup the system
    P   = SystemParameters(σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,num_gw_sources=num_gw_sources) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    init_parameters_optimal, priors_optimal = bilby_priors_dict(PTA,P,set_parameters_as_known=True)
    params = priors_optimal.sample(1)
    optimal_model_likelihood = KF.likelihood(params)
    logging.info(f"Ideal likelihood given optimal parameters = {optimal_model_likelihood}")


    
    #Now setup the actual prior, and run again to check everything OK
    init_parameters, priors = bilby_priors_dict(PTA,P,set_parameters_as_known=False)
    logging.info("Testing KF using parameters sampled from prior")
    params = priors.sample(1)
    model_likelihood = KF.likelihood(params)
    logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    #How many parameters are we trying to infer?
    logging.info(f"Total number of parameters to be inferred = {len(params)}")
    #...set the number of live points accordingly
    npoints = len(params) * 3
    if npoints < 1000:
        npoints=1000
    logging.info(f"Number of live points is {npoints}")


    logging.info("Selected injection parameters are as follows")

    # Now run the Bilby sampler
    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/",npoints=npoints)
    logging.info("The run has completed OK")
