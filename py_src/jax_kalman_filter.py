


import jax.numpy as np
from jax import lax
from jax import jit 
import sys 
import jax

"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
@jit
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)    
    slogdet = np.sum(np.log(S)) # Uses log rules and diagonality of covariance "matrix"
    value = -0.5*(slogdet+innovation @ x + N*np.log(2*np.pi))
    return value


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
@jit
def update(x, P, observation,R,Xfactor,ephemeris):


    H = 1.0 - Xfactor

    y_predicted = H*x - Xfactor*ephemeris
    y    = observation - y_predicted
    S    = H*P*H + R 
    K    = P*H/S 
    xnew = x + K*y


    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I_KH = 1.0 - K*H
    Pnew = I_KH * P * I_KH + K * R * K

    
    #And get the likelihood
    likelihood_value = log_likelihood(S,y)


    #and map the state to measurement space for plotting
    y_return = H*xnew - Xfactor*ephemeris

    return xnew, Pnew,likelihood_value,y_return


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
@jit
def predict(x,P,F,Q): 
    xp = F*x
    Pp = F*P*F + Q  

    return xp,Pp


from gravitational_waves import gw_psr_terms  

"""
Given some parameters, define all the Kalman matrices
"""
@jit
def setup_kalman_machinery(unknown_parameters_dict, known_parameters_dict):
    jax.debug.print("Welcome to the Kalman filter")

    #print("Welcome to the Kalman filter", P.Ω)

    #Extract parameters manually
    gamma   = known_parameters_dict["γ"]
    dt      = PTA.dt
    t       = PTA.t




    sigma_p = PTA.σp
    sigma_m = PTA.σm
 

    #State evolution matrices
    F = np.exp(-gamma*dt)
    Q = sigma_p**2 * (1. - np.exp(-2.0*gamma* dt)) / (2.0 * gamma)
    R = np.float64(sigma_m**2)

    # #measurement
    # X_factor = gw_psr_terms(P.δ,
    #                         P.α,
    #                         P.ψ,
    #                         PTA.q,
    #                         PTA.q_products,
    #                         P.h,
    #                         P.ι,
    #                         P.Ω,
    #                         PTA.t,
    #                         P.Φ0,
    #                         PTA.chi
    #                         )


    # f_EM = PTA.f + np.outer(PTA.t,PTA.fdot) #ephemeris correction


    #return F, Q, R,X_factor,f_EM
    return F



def kalman_filter(y, F, Q, R, H_fn,f_EM,initial_x, initial_P):



    def body(carry, t):
        x_hat_tm1, P_tm1,ll,y_hat = carry

        #Get the measurement matrix, the control vector matrix and the observation matrix at this iteration
        H_t = H_fn[t]
        T_t = f_EM[t]
        y_t = y[t]

        # Predict step 
        xp,Pp = predict(x_hat_tm1,P_tm1,F,Q)

        # Update step
        x_hat,P,ll_new,y_hat = update(xp, Pp, y_t,R,H_t,T_t)
        
        return (x_hat, P,ll+ll_new,y_hat), (x_hat, P,ll+ll_new,y_hat)


    # #State dimensions
    n_obs, n_dim = y.shape

    # # Initialize state estimates
    x_hat0 = np.zeros((n_dim,))
    x_hat0 = x_hat0.at[...].set(initial_x)

    P0 = np.zeros((n_dim,))
    P0 = P0.at[...].set(initial_P)
    
    y_hat0 = np.zeros_like(x_hat0)

    #Perform a single initial update step
   
    x_hat,P,ll,y_hat = update(x_hat0, P0, y[0],R,H_fn[0],f_EM[0])
   

    #Assign to variables
    #Is this necessary? Maybe to save first step?
    x_hat0 = x_hat0.at[...].set(x_hat)
    P0 = P0.at[...].set(P)
    y_hat0 = y_hat0.at[...].set(y_hat)
    

    # Now iterate over observations using scan
    #jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1)

    _, (x_hat, P,log_likelihood,y_hat) = lax.scan(f = body, init = (x_hat, P,ll,y_hat), xs = np.arange(1, n_obs))

    #print("scan complete")
    #print(y_hat.shape)

    # Prepend initial state estimate and error covariance
    x_hat_out = np.concatenate((x_hat0[np.newaxis, :], x_hat), axis=0)
    P_out     = np.concatenate((P0[np.newaxis, :], P), axis=0)
    y_hat_out = np.concatenate((y_hat0[np.newaxis, :], y_hat), axis=0)

    ll_return = log_likelihood[-1] #this is a cumsum so we only need last 
 

    return x_hat_out, P_out,ll_return,y_hat_out


