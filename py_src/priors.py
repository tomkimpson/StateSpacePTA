


import bilby
import numpy as np
import logging 
logging.getLogger().setLevel(logging.INFO)


"""
Add  constant prior vector
"""
def _add_to_bibly_priors_dict_constant(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = f
        i+= 1

    return init_parameters,priors



"""
Add  logarithmic prior vector
"""
def _add_to_bibly_priors_dict_log(x,label,init_parameters,priors,lower,upper): #same lower/upper for every one
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.LogUniform(lower,upper, key)
        
        i+= 1

    return init_parameters,priors


"""
Add uniform prior vector
"""
def _add_to_bibly_priors_dict_uniform(x,label,init_parameters,priors,tol):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(f-np.abs(f*tol),f+ np.abs(f*tol), key)
        
        i+= 1

    return init_parameters,priors



"""
Add uniform prior vector specifically for the chi vector
"""
def _add_to_bibly_priors_dict_chi(x,label,init_parameters,priors,k):
    
    i = 0
    for f in x:
        key = label+str(i)+f'_{k}'
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(0.0,2*np.pi, key)
        
        i+= 1

    return init_parameters,priors


"""
Add delta prior vector specifically for the chi vector
"""
def _add_to_bibly_priors_dict_chi_constant(x,label,init_parameters,priors,k):
    
    i = 0
    for f in x:
        key = label+str(i)+f'_{k}'
        init_parameters[key] = None

        priors[key] = f
        
        i+= 1

    return init_parameters,priors

"""
Set a prior on the state parameters
"""
def _set_prior_on_state_parameters(init_parameters,priors,f,fdot,σp,γ,d,chi,K,set_parameters_as_known,measurement_model):

    if set_parameters_as_known:
        logging.info('Setting fully informative priors on PSR parameters')

        init_parameters,priors = _add_to_bibly_priors_dict_constant(f,"f0",init_parameters,priors)     
        init_parameters,priors = _add_to_bibly_priors_dict_constant(fdot,"fdot",init_parameters,priors)    
        init_parameters,priors = _add_to_bibly_priors_dict_constant(σp,"sigma_p",init_parameters,priors)  
        init_parameters,priors = _add_to_bibly_priors_dict_constant(γ,"gamma",init_parameters,priors)           
        for k in range(K):
            init_parameters,priors = _add_to_bibly_priors_dict_chi_constant(chi[k,:],"chi",init_parameters,priors,k) 
    
    else:
        logging.info('Setting uninformative priors on PSR parameters')

        init_parameters,priors = _add_to_bibly_priors_dict_uniform(f,"f0",init_parameters,priors,tol=1e-10)      #uniform
        init_parameters,priors = _add_to_bibly_priors_dict_uniform(fdot,"fdot",init_parameters,priors,tol=0.01) #uniform

        #If we set the true process noise to zero, then don't bother searching over this parameter
        if np.all(σp) == 0.0:
            logging.info('The true process noise is zero.')
            logging.info('Not setting a prior for σp') 
            init_parameters,priors = _add_to_bibly_priors_dict_constant(σp,"sigma_p",init_parameters,priors)           #constant
        else:
            init_parameters,priors = _add_to_bibly_priors_dict_log(σp,"sigma_p",init_parameters,priors,1e-21,1e-19) #log. 
        
        
        init_parameters,priors = _add_to_bibly_priors_dict_constant(γ,"gamma",init_parameters,priors)           # constant
        

        if (measurement_model == 'null') or (measurement_model == 'earth') :
            for k in range(K):
                init_parameters,priors = _add_to_bibly_priors_dict_chi_constant(chi[k,:],"chi",init_parameters,priors,k) #dont need prior on chi for the null model or the earth terms model

        else:
            for k in range(K):
                init_parameters,priors = _add_to_bibly_priors_dict_chi(chi[k,:],"chi",init_parameters,priors,k) #uniform


    return init_parameters,priors 

"""
Set a prior on the measurement parameters
"""
def _set_prior_on_measurement_parameters(init_parameters,priors,P,set_parameters_as_known):

    if set_parameters_as_known: #don't set a prior, just assume these are known exactly a priori

        logging.info('Setting fully informative priors on GW parameters')
        for k in range(P.num_gw_sources):
            #Add all the GW quantities
            init_parameters[f"omega_gw_{k}"] = None
            priors[f"omega_gw_{k}"] = P.Ω[k]

            init_parameters[f"phi0_gw_{k}"] = None
            priors[f"phi0_gw_{k}"] = P.Φ0[k]

            init_parameters[f"psi_gw_{k}"] = None
            priors[f"psi_gw_{k}"] = P.ψ[k]

            init_parameters[f"iota_gw_{k}"] = None
            priors[f"iota_gw_{k}"] = P.ι[k]

            init_parameters[f"delta_gw_{k}"] = None
            priors[f"delta_gw_{k}"] = P.δ[k]

            init_parameters[f"alpha_gw_{k}"] = None
            priors[f"alpha_gw_{k}"] = P.α[k]

            init_parameters[f"h_{k}"] = None
            priors[f"h_{k}"] = P.h[k]

    else:
        logging.info('Setting uninformative priors on GW parameters')


        if P.measurement_model == "null":

            for k in range(P.num_gw_sources): #dont need a prior on the GW parameters for the null model.
        
                #Add all the GW quantities
                init_parameters[f"omega_gw_{k}"] = None
                priors[f"omega_gw_{k}"] = 1.0

                init_parameters[f"phi0_gw_{k}"] = None
                priors[f"phi0_gw_{k}"] = 1.0

                init_parameters[f"psi_gw_{k}"] = None
                priors[f"psi_gw_{k}"] = 1.0

                init_parameters[f"iota_gw_{k}"] = None
                priors[f"iota_gw_{k}"] = 1.0

                init_parameters[f"delta_gw_{k}"] = None
                priors[f"delta_gw_{k}"] = 1.0

                init_parameters[f"alpha_gw_{k}"] = None
                priors[f"alpha_gw_{k}"] = 1.0

                init_parameters[f"h_{k}"] = None
                priors[f"h_{k}"] = 1.0


        else:

            for k in range(P.num_gw_sources):
            
                #Add all the GW quantities
                init_parameters[f"omega_gw_{k}"] = None
                priors[f"omega_gw_{k}"] = bilby.core.prior.Uniform(1e-7, 9e-7, 'omega_gw')


                init_parameters[f"phi0_gw_{k}"] = None
                priors[f"phi0_gw_{k}"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'phi0_gw')

                init_parameters[f"psi_gw_{k}"] = None
                priors[f"psi_gw_{k}"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')

                init_parameters[f"iota_gw_{k}"] = None
                priors[f"iota_gw_{k}"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'iota_gw')

                init_parameters[f"delta_gw_{k}"] = None
                priors[f"delta_gw_{k}"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'delta_gw')

                init_parameters[f"alpha_gw_{k}"] = None
                priors[f"alpha_gw_{k}"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')

                init_parameters[f"h_{k}"] = None
                #priors[f"h_{k}"] = bilby.core.prior.LogUniform(1e-16, 5e-15, 'h')
                priors[f"h_{k}"] = bilby.core.prior.Uniform(1e-15, 9e-15, 'h')
                #priors[f"h_{k}"] = bilby.core.prior.LogUniform(1e-14, 9e-14, 'h')


    return init_parameters,priors 


"""
Main external function for defining priors. c.f. https://arxiv.org/pdf/2008.12320.pdf
"""
def bilby_priors_dict(PTA,P,set_state_parameters_as_known=False,set_measurement_parameters_as_known=False):


    logging.info('Setting the bilby priors dict')


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()
    
    #Measurement priors
    init_parameters,priors = _set_prior_on_measurement_parameters(init_parameters,priors,P,set_measurement_parameters_as_known) 

    #State priors
    init_parameters,priors = _set_prior_on_state_parameters(init_parameters,priors,PTA.f,PTA.fdot,PTA.σp,PTA.γ,PTA.d,PTA.chi,P.num_gw_sources,set_state_parameters_as_known,P.measurement_model)

    #Measurement noise priors. Always known
    init_parameters["sigma_m"] = None
    priors["sigma_m"] = P.σm



    return init_parameters,priors


