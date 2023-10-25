


import jax.numpy as np
from jax import lax

"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
# @njit(fastmath=True)
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)    
    slogdet = np.sum(np.log(S)) # Uses log rules and diagonality of covariance "matrix"
    value = -0.5*(slogdet+innovation @ x + N*np.log(2*np.pi))
    return value


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""

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
    
    return xnew, Pnew,likelihood_value #,y_predicted


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""

def predict(x,P,F,Q): 
    xp = F*x
    Pp = F*P*F + Q  

    return xp,Pp


from gravitational_waves import gw_psr_terms  

"""
Given some parameters, define all the Kalman matrices
"""
def setup_kalman_machinery(P,PTA):


    #Extract parameters manually
    gamma   = PTA.γ
    dt      = PTA.dt


    # f0      = PTA.f
    # fdot    = PTA.fdot
    t       = PTA.t
    sigma_p = PTA.σp
    sigma_m = PTA.σm
 





    #State evolution matrices
    F = np.exp(-gamma*dt)
    Q = sigma_p**2 * (1. - np.exp(-2.0*gamma* dt)) / (2.0 * gamma)
    R = np.float64(sigma_m**2)

    #measurement
    X_factor = gw_psr_terms(P.δ,
                            P.α,
                            P.ψ,
                            PTA.q,
                            PTA.q_products,
                            P.h,
                            P.ι,
                            P.Ω,
                            PTA.t,
                            P.Φ0,
                            PTA.chi
                            )


    f_EM = PTA.f + np.outer(PTA.t,PTA.fdot) #ephemeris correction

    #fdot_time =  np.outer(t,fdot) #This has shape(n times, n pulsars)
    #T_fn = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)

    #Process and measurement noise
    #Q = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
    #R = sigma_m**2





    # #Measurement matrix
    # H_fn = gw_prefactor_optimised(delta_gw,
    #                                                 alpha_gw,
    #                                                 psi_gw,
    #                                                 PTA.q,
    #                                                 PTA.q_products,
    #                                                 h,
    #                                                 iota_gw,
    #                                                 omega_gw,
    #                                                 d,
    #                                                 t,
    #                                                 phi0_gw
    #                                                 )




    return F, Q, R,X_factor,f_EM



def kalman_filter(y, F, Q, R, H_fn,f_EM,initial_x, initial_P):



    def body(carry, t):
        x_hat_tm1, P_tm1,ll = carry

        #Get the measurement matrix, the control vector matrix and the observation matrix at this iteration
        H_t = H_fn[t]
        T_t = f_EM[t]
        y_t = y[t]

        # Predict step 
        xp,Pp = predict(x_hat_tm1,P_tm1,F,Q)

        # Update step
        x_hat,P,ll = update(xp, Pp, y_t,R,H_t,T_t)
        
        return (x_hat, P,ll), (x_hat, P,ll)


    #State dimensions
    n_obs, n_dim = y.shape

    # Initialize state estimates
    x_hat0 = np.zeros((n_dim,))
    x_hat0 = x_hat0.at[...].set(initial_x)

    P0 = np.zeros((n_dim,))
    P0 = P0.at[...].set(initial_P)
    

    #Perform a single initial update step
    H_t = H_fn[0]
    T_t = f_EM[0]
    y_t = y[0]
    x_hat,P,ll = update(x_hat0, P0, y_t,R,H_t,T_t)

    #Assign to variabless
    x_hat0 = x_hat0.at[...].set(x_hat)
    P0 = P0.at[...].set(P)
    log_likelihood = np.float64(ll)
    


    # Now tterate over observations using scan
    _, (x_hat, P,log_likelihood) = lax.scan(body, (x_hat0, P0,log_likelihood), np.arange(1, n_obs))


    # Prepend initial state estimate and error covariance
    x_hat = np.concatenate((x_hat0[np.newaxis, :], x_hat), axis=0)
    P = np.concatenate((P0[np.newaxis, :], P), axis=0)

    return x_hat, P,log_likelihood


