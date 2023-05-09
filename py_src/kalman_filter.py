


import numpy as np 
#from gravitational_waves import gw_model

from numba import jit,config
from system_parameters import disable_JIT,heterodyne
config.DISABLE_JIT = disable_JIT

from model import F_function,T_function,R_function,Q_function #H function is defined via a class init

"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
@jit(nopython=True)
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)
    
    slogdet = np.sum(np.log(S)) # Uses log rules and diagonality of covariance "matrix"
    value = -0.5*(slogdet+innovation @ x + N*np.log(2*np.pi))
    return value
    #return -np.log(np.sum(np.abs(innovation)))
    #return -np.log(innovation @ innovation)


@jit(nopython=True)
def cauchy_likelihood(S,innovation):

    x = innovation / (0.8*np.sqrt(S))  
    N = len(x)

    slogdet = np.sum(np.log(0.8**2 * S))
    innovation_log = np.sum(np.log(1+x**2))
    value= -0.5 * (2*N*np.log(np.pi) +slogdet +2* innovation_log)
    return value 


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
@jit(nopython=True)
def update(x, P, observation,R,H,ephemeris):

    
    y    = observation - H*x + ephemeris

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
    l = log_likelihood(S,y)
    
    return xnew, Pnew,l


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
@jit(nopython=True)
def predict(x,P,F,T,Q): 
    xp = F*x + T 
    Pp = F*P*F + Q  
    return xp,Pp



class KalmanFilter:
    """
    A class to implement the Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

    """

    def __init__(self,Model, Observations,PTA):

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


        if heterodyne:
            print("Running the KF with heterodyned settings")
            self.ephemeris = PTA.ephemeris 
            self.x0 =  self.observations[0,:] + self.ephemeris[0,:] 

        else:
            print("Running the KF with non-heterodyned settings")
            self.ephemeris = np.zeros_like(PTA.ephemeris)
            self.x0 =  self.observations[0,:] 

    


    def likelihood(self,parameters):

        #Bilby takes a dict
        #For us this is annoying - map some quantities to be vectors
        f,fdot,gamma,d = map_dicts_to_vector(parameters)
        
        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        R = R_function(parameters["sigma_m"])
        Q = Q_function(gamma,parameters["sigma_p"],self.dt)
        F = F_function(gamma,self.dt)
        T = T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars

        #Initialise x and P
        x = self.x0 # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr) * 0.1 #Guess that the uncertainty is about 0.1 Hz 


        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
        modulation_factors = self.H_function(parameters["delta_gw"],
                                             parameters["alpha_gw"],
                                             parameters["psi_gw"],
                                             self.q,
                                             self.q_products,
                                             parameters["h"],
                                             parameters["iota_gw"],
                                             parameters["omega_gw"],
                                             d,
                                             self.t,
                                             parameters["phi0_gw"]
                                            )









        #Initialise the likelihood
        likelihood = 0.0
              

        x,P,l = update(x,P, self.observations[0,:],R,modulation_factors[0,:],self.ephemeris[0,:])
        likelihood +=l


        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr))
        y_results = np.zeros_like(x_results)
        x_results[0,:] = x
        y_results[0,:] = modulation_factors[0,:]*x - self.ephemeris[0,:]


        for i in np.arange(1,self.Nsteps):

            
            obs = self.observations[i,:]
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, obs,R,modulation_factors[i,:],self.ephemeris[i,:])
            likelihood +=l

            x_results[i,:] = x
            y_results[i,:] = modulation_factors[i,:]*x - self.ephemeris[i,:]


        return likelihood,x_results,y_results

     

"""
Useful function which maps repeated quantities in the dictionary to a
vector. Is there a more efficient way to do this? 
"""
def map_dicts_to_vector(parameters_dict):

    f = np.array([val for key, val in parameters_dict.items() if "f0" in key])
    fdot = np.array([val for key, val in parameters_dict.items() if "fdot" in key])
    gamma = np.array([val for key, val in parameters_dict.items() if "gamma" in key])
    d = np.array([val for key, val in parameters_dict.items() if "distance" in key])

    return f,fdot,gamma,d














#Scratch space
    #return -np.log(np.abs(value))
    #return -0.5*(innovation @ x + N*np.log(2*np.pi))

    #return -np.log(np.abs(innovation @ innovation))
    #return -np.sum(innovation**2)