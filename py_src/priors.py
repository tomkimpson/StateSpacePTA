


import bilby

import numpy as np
import random
import logging 
logging.getLogger().setLevel(logging.INFO)


def add_to_priors_dict(x,label,dict_A):



    i = 0
    for f in x:
        key = label+str(i)
        dict_A[key] = f
        i+= 1

    return dict_A



def add_to_priors_dict_erroneous(x,label,dict_A,tol):


    i = 0
    for f in x:
        value = random.uniform(f*(1-tol), f*(1+tol))
        key = label+str(i)
        dict_A[key] = value
        i+= 1

    return dict_A



def add_to_bibly_priors_dict_constant(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = f
        i+= 1

    return init_parameters,priors


def add_to_bibly_priors_dict_log(x,label,init_parameters,priors,tol):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.LogUniform(1e-21,1e-19, key)
        logging.info(f"Sigma p true value is {key} {f}")
        
        i+= 1

    return init_parameters,priors




def add_to_bibly_priors_dict(x,label,init_parameters,priors,tol):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(f-np.abs(f*tol),f+ np.abs(f*tol), key)
        
        i+= 1

    return init_parameters,priors



"""
Create a dict of parameters to be consumed by the Kalman likelihood function
"""
def priors_dict(pulsar_parameters,P):


   priors = dict({
               "omega_gw": P.Ω,
               "phi0_gw":P.Φ0,
               "psi_gw":P.ψ,
               "iota_gw":P.ι,
               "delta_gw":P.δ,
               "alpha_gw":P.α,
               "h": P.h})
   priors = add_to_priors_dict(pulsar_parameters.f,"f0",priors)
   priors = add_to_priors_dict(pulsar_parameters.fdot,"fdot",priors)
   priors = add_to_priors_dict(pulsar_parameters.d,"distance",priors)
   priors = add_to_priors_dict(pulsar_parameters.γ,"gamma",priors)
   priors = add_to_priors_dict(pulsar_parameters.σp,"sigma_p",priors)
   priors["sigma_m"]= pulsar_parameters.σm
  
   return priors




# https://arxiv.org/pdf/2008.12320.pdf
def bilby_priors_dict(PTA,P):


    logging.info('Using the default bilby priors dict')


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    #Add all the GW quantities
    init_parameters["omega_gw"] = None
    priors["omega_gw"] = bilby.core.prior.LogUniform(1e-9, 1e-5, 'omega_gw')


    init_parameters["phi0_gw"] = None
    priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'phi0_gw')

    init_parameters["psi_gw"] = None
    priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')

    init_parameters["iota_gw"] = None
    priors["iota_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'iota_gw')


    init_parameters["delta_gw"] = None
    priors["delta_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'delta_gw')


    init_parameters["alpha_gw"] = None
    priors["alpha_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')


    init_parameters["h"] = None
    #priors["h"] = bilby.core.prior.LogUniform(P["h"]/1e2, P["h"]*1e2, 'h') #prior on h is always 2 orders of magnitude either side of true value 
    priors["h"] = bilby.core.prior.LogUniform(1e-14, 1e-11, 'h')



    init_parameters,priors = add_to_bibly_priors_dict(PTA.f,"f0",init_parameters,priors,tol=0.01)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.fdot,"fdot",init_parameters,priors,tol=0.01)
    #init_parameters,priors = add_to_bibly_priors_dict(PTA.σp,"sigma_p",init_parameters,priors,tol=0.01)
    init_parameters,priors = add_to_bibly_priors_dict_log(PTA.σp,"sigma_p",init_parameters,priors,tol=0.01)
    


    #These guys are all constant     
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.d,"distance",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.γ,"gamma",init_parameters,priors)


    init_parameters["sigma_m"] = None
    priors["sigma_m"] = 1e-11



    return init_parameters,priors



def bilby_priors_dict_null(PTA,P):


    logging.info('Using the null bilby priors dict')

    init_parameters = {}
    priors = bilby.core.prior.PriorDict()



    init_parameters,priors = add_to_bibly_priors_dict(PTA.f,"f0",init_parameters,priors,tol=0.01)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.fdot,"fdot",init_parameters,priors,tol=0.01)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.σp,"sigma_p",init_parameters,priors,tol=0.01)
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.γ,"gamma",init_parameters,priors)


    init_parameters["sigma_m"] = None
    priors["sigma_m"] = 1e-11


    #We define the GW parameters for consistency but these are not actually used
    #again a bit hacky. Will need to clear this up, but doing it this way
    #lets us have a single H_function() type call
    init_parameters["omega_gw"] = None
    priors["omega_gw"] = 1.0


    init_parameters["phi0_gw"] = None
    priors["phi0_gw"] = 1.0  

    init_parameters["psi_gw"] = None
    priors["psi_gw"] = 1.0

    init_parameters["iota_gw"] = None
    priors["iota_gw"] =1.0 


    init_parameters["delta_gw"] = None
    priors["delta_gw"] =1.0


    init_parameters["alpha_gw"] = None
    priors["alpha_gw"] = 1.0


    init_parameters["h"] = None
    priors["h"] = 1.0


    #d can be undefined without any issues



    return init_parameters,priors


# https://arxiv.org/pdf/2008.12320.pdf
def bilby_priors_dict_earth(PTA,P):
    
    logging.info('Using the earth bilby priors dict')

    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    #Add all the GW quantities
    init_parameters["omega_gw"] = None
    priors["omega_gw"] = bilby.core.prior.LogUniform(1e-9, 1e-5, 'omega_gw')


    init_parameters["phi0_gw"] = None
    priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'phi0_gw')

    init_parameters["psi_gw"] = None
    priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')

    init_parameters["iota_gw"] = None
    priors["iota_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'iota_gw')


    init_parameters["delta_gw"] = None
    priors["delta_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'delta_gw')


    init_parameters["alpha_gw"] = None
    priors["alpha_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')


    init_parameters["h"] = None
    priors["h"] = bilby.core.prior.LogUniform(1e-14, 1e-11, 'h')



    init_parameters,priors = add_to_bibly_priors_dict(PTA.f,"f0",init_parameters,priors,tol=0.1)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.fdot,"fdot",init_parameters,priors,tol=0.1)
    #init_parameters,priors = add_to_bibly_priors_dict(PTA.σp,"sigma_p",init_parameters,priors,tol=0.01)
    init_parameters,priors = add_to_bibly_priors_dict_log(PTA.σp,"sigma_p",init_parameters,priors,tol=0.01)
    #These guys are all constant     
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.γ,"gamma",init_parameters,priors)


    init_parameters["sigma_m"] = None
    priors["sigma_m"] = 1e-11


    #distance d can be undefined without any issues


    return init_parameters,priors
