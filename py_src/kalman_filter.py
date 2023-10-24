import numpy as np 
#from numba import njit
from model import F_function,R_function,Q_function # H function is defined via a class init


"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
#@njit(fastmath=True)
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)    
    slogdet = np.sum(np.log(S)) # Uses log rules and diagonality of covariance "matrix"
    value = -0.5*(slogdet+innovation @ x + N*np.log(2*np.pi))
    return value


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
#@njit(fastmath=True)
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
    
    return xnew, Pnew,likelihood_value,y_predicted


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
#@njit(fastmath=True)
def predict(x,P,F,Q): 
    xp = F*x
    Pp = F*P*F + Q  

    return xp,Pp



"""
Given a Bilby dict, make it a numpy array 
"""
def dict_to_array(some_dict,target_keys):
    selected_dict = {k:some_dict[k] for k in target_keys}
    return np.array(list(selected_dict.values())).flatten()




class KalmanFilter:
    """
    A class to implement the Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

    """

    def __init__(self,Model, Observations,PTA,P):

        """
        Initialize the class. 
        """

        self.model = Model
        self.observations = Observations
        self.dt = PTA.dt
        self.q = PTA.q
        self.t = PTA.t
        self.q = PTA.q
        self.q_products = PTA.q_products

        self.Npsr = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]
        self.NF = PTA.NF
        self.H_function = Model.H_function
        self.x0 =  self.observations[0,:] 
        self.num_gw_sources = Model.num_gw_sources

        #Define some arrays of strings
        #These are the names of the static parameters
        #We will use these to parse the bilby dict into arrays.
        
        ## length num_gw_sources
        #self.list_of_omega_keys = [f'omega_gw_{i}' for i in range(self.num_gw_sources)]
        #self.list_of_phi0_keys  = [f'phi0_gw_{i}' for i in range(self.num_gw_sources)]
        #self.list_of_psi_keys   = [f'psi_gw_{i}' for i in range(self.num_gw_sources)]
        #self.list_of_iota_keys  = [f'iota_gw_{i}' for i in range(self.num_gw_sources)]
        #self.list_of_delta_keys = [f'delta_gw_{i}' for i in range(self.num_gw_sources)]
        #self.list_of_alpha_keys = [f'alpha_gw_{i}' for i in range(self.num_gw_sources)]
        #self.list_of_h_keys     = [f'h_{i}' for i in range(self.num_gw_sources)]

        ## length Npsr
        #self.list_of_f_keys        = [f'f0{i}' for i in range(self.Npsr)]
        #self.list_of_fdot_keys     = [f'fdot{i}' for i in range(self.Npsr)]
        #self.list_of_gamma_keys    = [f'gamma{i}' for i in range(self.Npsr)]
        #self.list_of_sigma_p_keys  = [f'sigma_p{i}' for i in range(self.Npsr)]

        # length Npsr x num_gw_sources
        #self.list_of_chi_keys = [ [f'chi{i}_{k}' for i in range(self.Npsr)] for k in range(self.num_gw_sources)]



        self.phi0_gw  = P.Φ0
        self.psi_gw   = P.ψ
        self.iota_gw  = P.ι
        self.delta_gw = P.δ
        self.alpha_gw = P.α
        self.h     = P.h

        self.f = PTA.f 
        self.fdot = PTA.fdot
        self.gamma = PTA.γ
        self.chi = PTA.chi
        self.sigma_p = PTA.σp

        self.sigma_m = P.σm




    # def parse_dictionary(self,parameters_dict):
        
        
    #     omega_gw = dict_to_array(parameters_dict,self.list_of_omega_keys)
    #     phi0_gw  = dict_to_array(parameters_dict,self.list_of_phi0_keys)
    #     psi_gw   = dict_to_array(parameters_dict,self.list_of_psi_keys)
    #     iota_gw  = dict_to_array(parameters_dict,self.list_of_iota_keys)
    #     delta_gw = dict_to_array(parameters_dict,self.list_of_delta_keys)
    #     alpha_gw = dict_to_array(parameters_dict,self.list_of_alpha_keys)
    #     h        = dict_to_array(parameters_dict,self.list_of_h_keys)

    #     # Pulsar paramters
    #     f       = dict_to_array(parameters_dict,self.list_of_f_keys)
    #     fdot    = dict_to_array(parameters_dict,self.list_of_fdot_keys)
    #     gamma   = dict_to_array(parameters_dict,self.list_of_gamma_keys)
    #     sigma_p = dict_to_array(parameters_dict,self.list_of_sigma_p_keys)

    #     chi = np.asarray([dict_to_array(parameters_dict,self.list_of_chi_keys[k]) for k in range(self.num_gw_sources)])


    #     #Other noise parameters
    #     sigma_m = parameters_dict["sigma_m"]

      
    #     return omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,f,fdot,gamma,sigma_p,chi,sigma_m



    def likelihood(self,parameters):

        #Map from the dictionary into variables and arrays
        #omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,f,fdot,gamma,sigma_p,chi,sigma_m = self.parse_dictionary(parameters)
        #print("The parameters are:")
        #print(parameters)

        omega_gw,phi0_gw = np.array([parameters[0]]),np.array([parameters[1]])

    
        #phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,f,fdot,gamma,sigma_p,chi,sigma_m = self.phi0_gw,self.psi_gw,self.iota_gw,self.delta_gw,self.alpha_gw,self.h,self.f,self.fdot,self.gamma,self.sigma_p,self.chi,self.sigma_m
        psi_gw,iota_gw,delta_gw,alpha_gw,h,f,fdot,gamma,sigma_p,chi,sigma_m = self.psi_gw,self.iota_gw,self.delta_gw,self.alpha_gw,self.h,self.f,self.fdot,self.gamma,self.sigma_p,self.chi,self.sigma_m


        #Precompute transition/Q/R Kalman matrices
        #F,Q,R are time-independent functions of the parameters
        F = F_function(gamma,self.dt)
        R = R_function(sigma_m)
        Q = Q_function(gamma,sigma_p,self.dt)
     
        #Initialise x and P
        x = self.x0 # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr)* sigma_m * 1e3 #Guess that the uncertainty in the initial state is a few orders of magnitude greater than the measurement noise


        # Precompute the influence of the GW
        # This is solely a function of the parameters and the t-variable but NOT the states
        X_factor = self.H_function(delta_gw,
                                   alpha_gw,
                                   psi_gw,
                                   self.q,
                                   self.q_products,
                                   h,
                                   iota_gw,
                                   omega_gw,
                                   self.t,
                                   phi0_gw,
                                   chi
                                )
        
        #Define an ephemeris correction
        f_EM = f + np.outer(self.t,fdot) #ephemeris correction


        #Initialise the likelihood
        likelihood = 0.0
              
       
        #Do the first update step
        x,P,likelihood_value,y_predicted = update(x,P, self.observations[0,:],R,X_factor[0,:],f_EM[0,:])
        likelihood +=likelihood_value

        #Don't bother storing results of the state. We just want the likelihoods
        for i in np.arange(1,self.Nsteps):
             obs                              = self.observations[i,:]                                     #The observation at this timestep
             x_predict, P_predict             = predict(x,P,F,Q)                                           #The predict step
             x,P,likelihood_value,y_predicted = update(x_predict,P_predict, obs,R,X_factor[i,:],f_EM[i,:]) #The update step    
             likelihood +=likelihood_value

        return likelihood







