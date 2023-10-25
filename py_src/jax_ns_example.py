
from jax.config import config
config.update("jax_enable_x64", True)

# JAX
import jax.numpy as np
from jax import random
from jax import lax

#jaxns
from jaxns import Prior, Model
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
tfb = tfp.bijectors



from numpyro.distributions import LogUniform


import matplotlib.pyplot as plt
from gravitational_waves import gw_prefactor_optimised
from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from copy import deepcopy





#### a playground script exploring the use of jaxns https://jaxns.readthedocs.io/en/latest/# for the Kalman likelihood inference



### Specify the parameters and create some synthetic data

P    = SystemParameters()          # define the system parameters as a class
PTA  = Pulsars(P)              # setup the PTA
data = SyntheticData(PTA,P)    # generate some synthetic data

y = data.f_measured # This is the data we will run our likelihood on. It has shape (ntimes x N pulsars)



### Define all the Kalman -machinery, matrices



"""
Given some parameters, define all the Kalman matrices
"""
def setup_kalman_machinery(P,PTA):


    #Extract parameters manually
    gamma   = PTA.gamma
    dt      = PTA.dt
    f0      = PTA.f
    fdot    = PTA.fdot
    t       = PTA.t
    sigma_p = PTA.sigma_p
    sigma_m = PTA.sigma_m
    d       = PTA.d

    omega_gw =P.omega_gw
    phi0_gw  =P.phi0_gw
    psi_gw   =P.psi_gw
    iota_gw  =P.iota_gw
    delta_gw =P.delta_gw
    alpha_gw =P.alpha_gw
    h        =P.h



    #State evolution matrices
    F = np.exp(-gamma*dt)
    fdot_time =  np.outer(t,fdot) #This has shape(n times, n pulsars)
    T_fn = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)

    #Process and measurement noise
    Q = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
    R = sigma_m**2





    #Measurement matrix
    H_fn = gw_prefactor_optimised(delta_gw,
                                                    alpha_gw,
                                                    psi_gw,
                                                    PTA.q,
                                                    PTA.q_products,
                                                    h,
                                                    iota_gw,
                                                    omega_gw,
                                                    d,
                                                    t,
                                                    phi0_gw
                                                    )










    return F, Q, R,H_fn,T_fn




"""
The likelihood step of the Kalman filter
"""
def log_likelihood_fn(innovation,S):
    x = innovation / S 
    N = len(x)
    return -0.5*(np.dot(innovation,x) + N*np.log(2*np.pi)) #temporarily dropped slogdet for now




"""
The update step of the Kalman filter
"""
def update(x, P, observation,R,H):

    innovation = observation - H * x
    S          = H * P * H + R
    K          = P * H/S

    xnew = x + K*innovation


    I_KH = 1.0 - K*H
    Pnew = I_KH * P * I_KH + K * R * K


    l = log_likelihood_fn(innovation,S)

    return xnew,Pnew,l



"""
The predict step of the Kalman filter
"""
def predict(x,P,F,T,Q): 
    xp = F*x + T 
    Pp = F*P*F + Q   
    return xp,Pp



"""
A jax-style Kalman filter function 
"""
def kalman_filter(y, F, Q, R, H_fn,T_fn,initial_x, initial_P):



    def body(carry, t):
        x_hat_tm1, P_tm1,ll = carry

        #Get the measurement matrix, the control vector matrix and the observation matrix at this iteration
        H_t = H_fn[t]
        T_t = T_fn[t]
        y_t = y[t]

        # Predict step 
        xp,Pp = predict(x_hat_tm1,P_tm1,F,T_t,Q)


        # Update step
        x_hat,P,ll = update(xp, Pp, y_t,R,H_t)
        
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
    y_t = y[0]
    x_hat,P,ll = update(x_hat0, P0, y_t,R,H_t)

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







#Run the kalman filter once for optimal parameters to check everything works as expected
F, Q, R,H_fn,T_fn = setup_kalman_machinery(P,PTA)
initial_x = y[0,:]
initial_P = np.ones(len(initial_x)) * PTA.sigma_m*1e10 


x_result,P_result,l_result = kalman_filter(y, F, Q, R, H_fn,T_fn,initial_x, initial_P)

from plotting import plot_all
#plot_all(PTA.t,data.intrinsic_frequency,data.f_measured,x_result,psr_index =1,savefig=None)








# Now lets try and do some nested sampling
# We will try to just infer the value the parameter omega_gw


"""
Function of one parameter
Classes of P,PTA and data y are inherited from the global scope
"""
def likelihood_function(omega,phi_0):

    #Create class "copies"
    P1 = deepcopy(P)
    PTA1 = deepcopy(PTA)

    print("oemga = ", omega)
    P1.omega_gw = omega #reassign the value of omega in this new class
    P1.phi0_gw = phi_0 #reassign the value of omega in this new class


    F, Q, R,H_fn,T_fn = setup_kalman_machinery(P1,PTA1)
    initial_x = y[0,:]
    initial_P = np.ones(len(initial_x)) * PTA1.sigma_m*1e10 


    x_result,P_result,l_result = kalman_filter(y, F, Q, R, H_fn,T_fn,initial_x, initial_P)



    value = np.sum(l_result)

    return value







# lower_bound = 1e-8
# upper_bound = 1e-6

# # Define the log-uniform distribution using TFP's TransformedDistribution class
# log_uniform = tfp.distributions.TransformedDistribution(
#     distribution=tfp.distributions.Uniform(np.log(lower_bound), np.log(upper_bound)),
#     bijector=tfp.bijectors.Exp())



# import seaborn as sns 

# samples = log_uniform.sample(10000, seed=random.PRNGKey(0))
# #print(samples)
# log_samples = np.log(samples)
# sns.distplot(log_samples)
# #plt.xscale('log')
# plt.show()
# sys.exit()




# xx = np.logspace(-8,-6,int(1e3))
# yy = []
# for x in xx:
#     lval = likelihood_function(x)
#     yy.extend([lval])


# plt.plot(xx,yy)
# plt.xscale('log')
# plt.show()

#sys.exit()










"""
The prior on omega
"""
def prior_model():
    #omega = yield Prior(tfpd.Uniform(low=1e-9, high = 1e-6), name='omega')
    
    
    phi_0 = yield Prior(tfpd.Uniform(low=0.0, high = np.pi), name='phi_0')
 



    
    lower_bound = 1e-8
    upper_bound = 1e-6

    # Define the log-uniform distribution using TFP's TransformedDistribution class
    log_uniform = tfp.distributions.TransformedDistribution(
        distribution=tfp.distributions.Uniform(np.log(lower_bound), np.log(upper_bound)),
        bijector=tfp.bijectors.Exp())



    omega = yield Prior(log_uniform, name='omega')





    return omega,phi_0






model = Model(prior_model=prior_model, log_likelihood=likelihood_function)
print("Perform a sanity check")
model.sanity_check(random.PRNGKey(0), S=100)





#Now run jaxns


from jaxns import ExactNestedSampler
from jaxns import TerminationCondition

print("--------------Attepting to run nested sampler-----------------")

# Create the nested sampler class. In this case without any tuning.
ns = exact_ns = ExactNestedSampler(model=model, 
                                   num_live_points=1000, num_parallel_samplers=1,
                                   max_samples=1e4)

# ns = exact_ns = ExactNestedSampler(model=model, 
#                                    num_live_points=1000, num_parallel_samplers=2,max_samples=1e4)


termination_reason, state = exact_ns(random.PRNGKey(42),
                                     term_cond=TerminationCondition(live_evidence_frac=1e-1))
results = exact_ns.to_results(state, termination_reason)

print("COMPLETED")
print(exact_ns.summary(results))
exact_ns.plot_diagnostics(results)
print("Attempting a save")
exact_ns.save_results(results=results,save_file="test_save.npz")
exact_ns.plot_cornerplot(results)




