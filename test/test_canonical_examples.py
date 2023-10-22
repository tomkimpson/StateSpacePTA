


from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter
from priors import bilby_priors_dict
import numpy as np 


def test_canonical_example():

    #Parameters 
    h = 5e-15
    measurement_model = 'pulsar'
    seed = 1237 

    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #We do this as standard to allow JIT precompile
    init_parameters_optimal, priors_optimal = bilby_priors_dict(PTA,P,set_parameters_as_known=True)
    optimal_parameters = priors_optimal.sample(1)


   
    model_likelihood = KF.likelihood(optimal_parameters)
    assert model_likelihood == 585736.3162773821 #magic number! Taken from the original code with 1 source. The multi-gw code should give the same answer in the num_gw_sources=1 limit



def test_multiple_gw_sources():

    #Parameters 
    h = 5e-15
    measurement_model = 'pulsar'
    seed = 1237 
    num_gw_sources = 10

    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,num_gw_sources=num_gw_sources) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #We do this as standard to allow JIT precompile
    init_parameters_optimal, priors_optimal = bilby_priors_dict(PTA,P,set_parameters_as_known=True)
    optimal_parameters = priors_optimal.sample(1)


    model_likelihood = KF.likelihood(optimal_parameters)


    assert ~np.isnan(model_likelihood) #always passes if the code completes and likelihood is reasonable