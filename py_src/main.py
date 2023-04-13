




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter


#jax/jaxns imports
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp, vmap
from jaxns import resample
from jaxns import Prior, Model
from jax.config import config
config.update("jax_enable_x64", True)
tfpd = tfp.distributions


P   = SystemParameters()       #define the system parameters as a class
PTA = Pulsars(P)               #setup the PTA
data = SyntheticData(PTA,P)    # generate some synthetic data


#Define the model 
#LinearModel = LinearModel

#Initialise the Kalman filter
KF = KalmanFilter(LinearModel,data.f_measured,PTA)



#parameters = priors_dict(PTA,P) #global defied



phi0_gw  =P["phi0_gw"]
psi_gw   =P["psi_gw"]
iota_gw  =P["iota_gw"]
delta_gw =P["delta_gw"]
alpha_gw =P["alpha_gw"]
h        =P["h"]
f = PTA.f
fdot = PTA.fdot
d = PTA.d
gamma = PTA.gamma
sigma_p= PTA.sigma_p
sigma_m= PTA.sigma_m


def prior_model():
    print("THIS IS THE PRIOR MODEL--------------------------------")
    omega = yield Prior(tfpd.Uniform(low=3e-7, high = 7e-7), name='omega')
    return omega


def log_likelihood(omega):


    print("THIS IS THE LOG LIKELIHOOD----------------------------------")

    value = KF.likelihood_test(omega,
                        phi0_gw,
                        psi_gw,
                        iota_gw,
                        delta_gw,
                        alpha_gw,
                        h,
                        f,
                        fdot,
                        gamma, 
                        d,
                        sigma_p,
                        sigma_m
                        )


    #value = KF.likelihood_test(omega)
    
    print("return value =", value)
    return value




# def log_likelihood(omega):
#     """
#     Poisson likelihood.
#     """

#     return omega - 5e-7






#log_likelihood(4e-7)



model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

print("Perform a sanity check")
model.sanity_check(random.PRNGKey(0), S=100)
















   
    # #Bilby 
    # init_parameters, priors = bilby_priors_dict(PTA,P)
    # BilbySampler(KF,init_parameters,priors,label="all_gw",outdir="../data/nested_sampling/")







