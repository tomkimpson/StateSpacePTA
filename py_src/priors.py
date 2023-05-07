


import bilby

import numpy as np
import random

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


def add_to_bibly_priors_dict(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = bilby.core.prior.Uniform(f*0.95,f*1.05, key)
        i+= 1

    return init_parameters,priors




def priors_dict(pulsar_parameters,P):


   priors = dict({
               "omega_gw": P["omega_gw"],
               "phi0_gw":P["phi0_gw"],
               "psi_gw":P["psi_gw"],
               "iota_gw":P["iota_gw"],
               "delta_gw":P["delta_gw"],
               "alpha_gw":P["alpha_gw"],
               "h": P["h"]})
   priors = add_to_priors_dict(pulsar_parameters.f,"f0",priors)
   priors = add_to_priors_dict(pulsar_parameters.fdot,"fdot",priors)
   priors = add_to_priors_dict(pulsar_parameters.d,"distance",priors)
   priors = add_to_priors_dict(pulsar_parameters.gamma,"gamma",priors)
   priors["sigma_p"]= pulsar_parameters.sigma_p
   priors["sigma_m"]= pulsar_parameters.sigma_m

  
   return priors



"""
Define the pulsar parameters as being slightly wrong from their true values by a random factor < tol
"""
def erroneous_priors_dict(pulsar_parameters,P,tol):


   priors = dict({
               "omega_gw": P["omega_gw"]*tol,
               "phi0_gw":P["phi0_gw"]*tol,
               "psi_gw":P["psi_gw"],
               "iota_gw":P["iota_gw"]*tol,
               "delta_gw":P["delta_gw"],
               "alpha_gw":P["alpha_gw"]*tol,
               "h": P["h"]})
   priors = add_to_priors_dict_erroneous(pulsar_parameters.f,"f0",priors,tol)
   priors = add_to_priors_dict_erroneous(pulsar_parameters.fdot,"fdot",priors,tol)
   priors = add_to_priors_dict(pulsar_parameters.d,"distance",priors)
   priors = add_to_priors_dict_erroneous(pulsar_parameters.gamma,"gamma",priors,tol)
   priors["sigma_p"]= pulsar_parameters.sigma_p
   priors["sigma_m"]= pulsar_parameters.sigma_m

  
   return priors








# https://arxiv.org/pdf/2008.12320.pdf
def bilby_priors_dict(PTA,P):

    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    #Add all the GW quantities
    init_parameters["omega_gw"] = None
    priors["omega_gw"] = bilby.core.prior.LogUniform(1e-9, 1e-5, 'omega_gw')
    #priors["omega_gw"] =P["omega_gw"]


    init_parameters["phi0_gw"] = None
    priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'phi0_gw')
    #priors["phi0_gw"] =P["phi0_gw"]

    init_parameters["psi_gw"] = None
    #priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw',boundary="periodic")
    priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')
    #priors["psi_gw"] =P["psi_gw"]

    init_parameters["iota_gw"] = None
    priors["iota_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'iota_gw')
    #priors["iota_gw"] = P["iota_gw"]


    init_parameters["delta_gw"] = None
    #priors["delta_gw"] = bilby.core.prior.Uniform(1e-2, 6.283185, 'delta_gw')
    priors["delta_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'delta_gw')
    #priors["delta_gw"] = P["delta_gw"]


    init_parameters["alpha_gw"] = None
    priors["alpha_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')
    #priors["alpha_gw"] = P["alpha_gw"]


    init_parameters["h"] = None
    priors["h"] = bilby.core.prior.LogUniform(1e-4, 1e-1, 'h')
    #priors["h"] = P["h"]



    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.f,"f0",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.fdot,"fdot",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.d,"distance",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict_constant(PTA.gamma,"gamma",init_parameters,priors)


    #Noises
    init_parameters["sigma_p"] = None
    priors["sigma_p"] = P["sigma_p"] 
    #priors["sigma_p"] = 1e-3


    init_parameters["sigma_m"] = None
    #priors["sigma_m"] = P["sigma_m"]
    priors["sigma_m"] = 1.0



    return init_parameters,priors
