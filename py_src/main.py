

#Local imports
from pytree import specify_system_parameters
from pulsars import create_pulsars
from synthetic_data import SyntheticData



#other imports



from jax_kalman_filter import setup_kalman_machinery,kalman_filter


import jax.numpy as np


from jax import config
config.update("jax_enable_x64", True) #use double, not single precision

import copy



from jaxns import Prior, Model



from jax import random







#Functional programming using JAX. 

from plotting import global_plot


import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions


# def prior_model():
#     x = yield Prior(tfpd.Uniform(low=1e-7 * np.ones(1), high=9e-7 * np.ones(1)), name='omega')
#     return x




if __name__=="__main__":



    P   = specify_system_parameters(h=1e-10,σp=1e-15,σm=1e-12,Npsr=20,cadence=0.5) 
    PTA = create_pulsars(P)                                       # setup the PTA

 
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    y = data.f_measured
    



    # Run the kalman filter once for optimal parameters to check everything works as expected

    #Separate input types

    unknown_parameters_dict = {}


    known_parameters_dict = {"γ" : PTA.γ,
                             "dt" : {TA.dt}
                               }




    F, Q, R,X_factor,f_EM = setup_kalman_machinery(P,PTA)
    initial_x = y[0,:]
    initial_P = np.ones(len(initial_x)) * PTA.σm*1e3 
    x_result,P_result,l_result,y_result = kalman_filter(y, F, Q, R, X_factor,f_EM,initial_x, initial_P)
    predictions = [x_result,y_result]

    global_plot(data,predictions)

    # def log_likelihood_wrapper(x):
        
    #     P_i = SystemParameters(h=1e-10,σp=1e-15,σm=1e-12,Npsr=20,cadence=0.5) #make a copy. This is a failure point
    #     P_i.Ω = x 

    #     #print(P_i)

    #     F, Q, R,X_factor,f_EM = setup_kalman_machinery(P_i,PTA)
    #     initial_x = y[0,:]
    #     initial_P = np.ones(len(initial_x)) * PTA.σm*1e3 
    #     x_result,P_result,l_result,y_result = kalman_filter(y, F, Q, R, X_factor,f_EM,initial_x, initial_P)
    #     return l_result





    # model = Model(prior_model=prior_model,log_likelihood=log_likelihood_wrapper)

    # model.sanity_check(random.PRNGKey(0), S=100)




    #Lets do some NS
    






    
    #plot_all(t,states,measurements,measurements_clean,predictions_x,predictions_y,psr_index,savefig=None):









# scratch space










  # #these should all be jax objects
    # print(type(y_jax))
    # print(type(F))
    # print(type(Q))
    # print(type(R))
    # print(type(X_factor))
    # print(type(f_EM))
    # print(type(initial_x))
    # print(type(initial_P))
