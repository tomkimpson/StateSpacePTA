

from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData


from jax_kalman_filter import setup_kalman_machinery,kalman_filter


import jax.numpy as np

#Functional programming using JAX. 


if __name__=="__main__":



    P   = SystemParameters(σp=None) 
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data
    y = data.f_measured
    print("got the synthetic data")
    y_jax = np.array(y)




#Run the kalman filter once for optimal parameters to check everything works as expected
        # F, Q, R,H_fn,T_fn = setup_kalman_machinery(P,PTA)

    F, Q, R,X_factor,f_EM = setup_kalman_machinery(P,PTA)



    initial_x = y[0,:]
    initial_P = np.ones(len(initial_x)) * PTA.σm*1e3 


    print(type(y_jax))
    print(type(F))
    print(type(Q))
    print(type(R))
    print(type(X_factor))
    print(type(f_EM))
    print(type(initial_x))
    print(type(initial_P))

   
    x_result,P_result,l_result = kalman_filter(y_jax, F, Q, R, X_factor,f_EM,initial_x, initial_P)



# x_hat, P,log_likelihood