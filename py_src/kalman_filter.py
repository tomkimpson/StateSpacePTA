


import numpy as np 
from gravitational_waves import gw_prefactor_optimised
from scipy.stats import multivariate_normal

from math import prod


import sys

from numba import jit 

import numba









# def get_vector_likelihood(self,S,innovation):
#     x = innovation / S 
#     slogdet = np.sum(np.log(S)) #Uses log rules and diagonality of covariance "matrix"
    
    
#     return -0.5*(slogdet+innovation @ x + self.Npsr*np.log(2*np.pi))




@jit(nopython=True)
def get_vector_likelihood(S,innovation,factor):
    x = innovation / S 
    slogdet = np.sum(np.log(S)) #Uses log rules and diagonality of covariance "matrix"
    
    #print("log of innovation covariance:", np.log(S))
    ff = len(innovation)*np.log(2*np.pi)


    print("Likelihood:",slogdet,innovation @ x, ff )
    slogdet=0.0
    return -0.5*(slogdet+innovation @ x + ff)


@jit(nopython=True)
def update(x, P, observation,R,H,factor):

 
    #print("H = ", H)
   # print("Pin = ", P)

    y = observation - H*x
    S = H*P*H + R
    K = P*H/S
    xnew = x + K*y

    #identity = np.ones(2)
    #I_KH = self.identity - K*H 
    #I_KH2 = identity - K*H 
    I_KH = 1.0 - K*H


    #print(I_KH2 == I_KH)
    #print(I_KH)

    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    # In practice for this pulsar problem I have also found this expression more numerically stable
    # ...despite the extra cost of operations
    #I_KH = I - (K*H)
    #Pnew = I_KH * P * I_KH + K * R * K
    Pnew = I_KH * P

        
   # print("P out = ", Pnew )
   #print("Ikh = ", I_KH)



    l = get_vector_likelihood(S,y,factor)


    #print("---------------------------------")

    return xnew, Pnew,l







@jit(nopython=True)
def predict(x,P,F,T,Q):
    xp = F*x + T 
    Pp = F*P*F + Q

    #print("P predict:",F*P*F)
    #print("Q =" , Q)

    #print("f =", F)
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
        self.observations = np.array(Observations)
        self.dt = PTA.dt
        self.q = PTA.q
        self.q_products = PTA.q_products #.T
        self.t = PTA.t

        self.Npsr = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]


        print("The number of pulsars is: ",self.Npsr)
        


        # #Initialise x and P
        self.x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        #self.P = np.eye(self.Npsr) * 1e5 #1e-13*1e9 
        self.P = np.ones(self.Npsr)*1e5

        self.identity = np.ones(self.Npsr)


        self.factor = self.Npsr*np.log(2*np.pi)





    def likelihood(self,parameters):
        

        #print(parameters)
        #Two different methods for mapping the parameters dict to a vector
        #Performance seems about the same
        #f,fdot,gamma,d = map_dicts_to_vector(parameters)
        f,fdot,gamma,d = map_dicts_to_vector2(parameters,self.Npsr)
       
        
        # #Setup Q and R matrices.
        # #These are time-independent functions of the parameters
        Q = self.model.Q_function(gamma,parameters["sigma_p"],self.dt)
        R = self.model.R_function(self.Npsr,parameters["sigma_m"])



        #Initial values for x and P
        x = self.x 
        P = self.P#*parameters["sigma_m"]

        #Compute quantities that depend on the system parameters but are constant in time       
        modulation_factors = gw_prefactor_optimised(parameters["delta_gw"],
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
 
    
        #Precompute all the transition and control matrices
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars
        


        #Initialise the likelihood
        likelihood = 0.0
        i = 0
        x,P,l= update(x,P, self.observations[i,:],R,modulation_factors[i,:],self.factor)
        likelihood +=l


   
        for i in np.arange(1,self.Nsteps):
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, self.observations[i,:],R,modulation_factors[i,:],self.factor)
            likelihood +=l

        return likelihood

      

    

    """
    Identical to the above likelihood() function, but also returns predictions for the sates
    This could be included in the above function with some `if` conditions,
    but I prefer to keep the likelihood function clean for use with Bilby/ Nested Sampling
    """
    def likelihood_and_states(self,parameters):
        

        #print(parameters)
        #Two different methods for mapping the parameters dict to a vector
        #Performance seems about the same
        #f,fdot,gamma,d = map_dicts_to_vector(parameters)
        f,fdot,gamma,d = map_dicts_to_vector2(parameters,self.Npsr)
       
        
        # #Setup Q and R matrices.
        # #These are time-independent functions of the parameters
        Q = self.model.Q_function(gamma,parameters["sigma_p"],self.dt)
        R = self.model.R_function(self.Npsr,parameters["sigma_m"])



        #Initial values for x and P
        x = self.x 
        P = self.P#*parameters["sigma_m"]

        #Compute quantities that depend on the system parameters but are constant in time       
        modulation_factors = gw_prefactor_optimised(parameters["delta_gw"],
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
 
    
        #Precompute all the transition and control matrices
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars
        


        #Initialise the likelihood
        likelihood = 0.0
        i = 0
        x,P,l= update(x,P, self.observations[i,:],R,modulation_factors[i,:],self.factor)
        likelihood +=l

        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr))
        x_results[0,:] = x


   
        for i in np.arange(1,self.Nsteps):
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, self.observations[i,:],R,modulation_factors[i,:],self.factor)
            likelihood +=l
            x_results[i,:] = x

        return likelihood,x_results

            









"""
Useful function which maps repeted quantities in the dictionary to a
vector
"""
def map_dicts_to_vector(parameters_dict):


    f = np.array([val for key, val in parameters_dict.items() if "f0" in key])
    fdot = np.array([val for key, val in parameters_dict.items() if "fdot" in key])
    gamma = np.array([val for key, val in parameters_dict.items() if "gamma" in key])
    d = np.array([val for key, val in parameters_dict.items() if "distance" in key])

    return f,fdot,gamma,d



def map_dicts_to_vector2(parameters_dict,N):

    #print("map_dicts_to_vector2")

    xx = np.array([*parameters_dict.values()])
    #print(parameters_dict)
    #print(xx)
    #print("-------------")

    buffer = 7
    f = xx[buffer:buffer+N]
    fdot = xx[buffer+N: buffer+2*N]
    d = xx[buffer+2*N: buffer+3*N]
    gamma = xx[buffer+3*N:buffer+4*N]


    return f,fdot,gamma,d



