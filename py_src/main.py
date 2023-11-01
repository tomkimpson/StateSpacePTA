

from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData


from jax_kalman_filter import setup_kalman_machinery,kalman_filter


import jax.numpy as np


from jax import config
config.update("jax_enable_x64", True) #use double, not single precision


#Functional programming using JAX. 

from plotting import global_plot

if __name__=="__main__":



    P   = SystemParameters(h=1e-10,σp=1e-15,σm=1e-12,Npsr=20,cadence=0.5) 
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    y = data.f_measured
    



    # Run the kalman filter once for optimal parameters to check everything works as expected


    F, Q, R,X_factor,f_EM = setup_kalman_machinery(P,PTA)

    initial_x = y[0,:]
    initial_P = np.ones(len(initial_x)) * PTA.σm*1e3 
    x_result,P_result,l_result,y_result = kalman_filter(y, F, Q, R, X_factor,f_EM,initial_x, initial_P)


    predictions = [x_result,y_result]

    global_plot(data,predictions)


    
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
