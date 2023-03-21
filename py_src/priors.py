


import bilby

import numpy as np

def add_to_priors_dict(x,label,dict_A):



    i = 0
    for f in x:
        key = label+str(i)
        dict_A[key] = f
        i+= 1

    return dict_A


"""
Add the X
"""
def add_to_bibly_priors_dict(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = f
        i+= 1

    return init_parameters,priors




def priors_dict(pulsar_parameters,GW_parameters):


   priors = dict({
               "omega_gw": GW_parameters.omega_gw,
               "phi0_gw":GW_parameters.phi0_gw,
               "psi_gw":GW_parameters.psi_gw,
               "iota_gw": GW_parameters.iota_gw,
               "delta_gw":GW_parameters.delta_gw,
               "alpha_gw":GW_parameters.alpha_gw,
               "h": GW_parameters.h,
               "sigma_p": pulsar_parameters.sigma_p,
               "sigma_m": pulsar_parameters.sigma_m})
   priors = add_to_priors_dict(pulsar_parameters.f,"f0",priors)
   priors = add_to_priors_dict(pulsar_parameters.fdot,"fdot",priors)
   priors = add_to_priors_dict(pulsar_parameters.d,"distance",priors)
   priors = add_to_priors_dict(pulsar_parameters.gamma,"gamma",priors)

   return priors


def bilby_priors_dict(PTA):

    init_parameters = {}
    priors = bilby.core.prior.PriorDict()

    #Add all the GW quantities
    init_parameters["omega_gw"] = None
    priors["omega_gw"] = bilby.core.prior.LogUniform(1e-9, 1e-5, name='omega_gw', latex_label=r'\omega')
  

    init_parameters["phi0_gw"] = None
    priors["phi0_gw"] = bilby.core.prior.Uniform(1e-1, 2*np.pi,  name='phi0_gw', latex_label=r'\Phi_0')
  

    init_parameters["psi_gw"] = None
    priors["psi_gw"] = bilby.core.prior.Uniform(1e-1, 2*np.pi,  name='psi_gw', latex_label=r'\psi')
  

    init_parameters["iota_gw"] = None
    priors["iota_gw"] = bilby.core.prior.Uniform(1e-1, 2*np.pi,  name='iota_gw', latex_label=r'\iota')

    init_parameters["delta_gw"] = None
    priors["delta_gw"] = bilby.core.prior.Uniform(1e-1, 2*np.pi,  name='delta_gw', latex_label=r'\delta')

    init_parameters["alpha_gw"] = None
    priors["alpha_gw"] = bilby.core.prior.Uniform(1e-1, 2*np.pi,  name='alpha_gw', latex_label=r'\alpha')


    init_parameters["h"] = None
    priors["h"] = bilby.core.prior.LogUniform(1e-4, 1e0,  name='h', latex_label=r'h')


    init_parameters,priors = add_to_bibly_priors_dict(PTA.f,"f0",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.fdot,"fdot",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.d,"distance",init_parameters,priors)
    init_parameters,priors = add_to_bibly_priors_dict(PTA.gamma,"gamma",init_parameters,priors)




    #Noises
    init_parameters["sigma_p"] = None
    priors["sigma_p"] = bilby.core.prior.LogUniform(1e-8, 1e-3,  name='sigma_p', latex_label=r'\sigma_p')

    init_parameters["sigma_m"] = None

    priors["sigma_m"] = 1e-8


    return init_parameters,priors
