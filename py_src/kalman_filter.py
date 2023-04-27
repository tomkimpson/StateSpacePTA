


import numpy as np 
#from gravitational_waves import gw_model

#from numba import jit,config
#from system_parameters import disable_JIT
#config.DISABLE_JIT = disable_JIT

import sys

"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
# @jit(nopython=True)
def log_likelihood(S,innovation):
    x = np.linalg.solve(S,innovation)
    N = len(x)
    return -0.5*(np.linalg.slogdet(S)[-1] + innovation @ x + N*np.log(2*np.pi))

"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
# @jit(nopython=True)




"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
# @jit(nopython=True)




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


        self.f = PTA.f
        self.fdot = PTA.fdot
        self.gamma = PTA.gamma
        self.sigma_p = PTA.sigma_p
        self.sigma_m = PTA.sigma_m




    def update(self,x,P,observation):

        #print("This is the update step")
        hx = self.model.h_function(x,self.Ai,self.phase_i)
        H = self.model.H_function(x,self.Ai,self.phase_i)

       
        #print("x in:", x)
        
        y    = observation - hx
        #print(y)
        S    = H@P@H.T + self.R  
        K    = P@H.T@np.linalg.inv(S) 

        xnew = x + K@y

        #print("xout:")
        #print(xnew)
        #print("------------------")


        #Update the covariance 
        #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
        # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = np.eye(len(xnew)) - K@H
        Pnew = I_KH @ P @ I_KH.T + K @ self.R @ K.T
    
    
        #And get the likelihood
        l = log_likelihood(S,y)


        #print("x out:", xnew)


        #print("-------------------")

        
        return xnew, Pnew,l




    def predict(self,x,P,t): 
        T = self.model.T_function(self.f,self.fdot,self.gamma,t,self.dt) 

        #print("This is the predict step")
        
        xp = self.F@x + T 
        Pp = self.F@P@self.F.T + self.Q   

       
        #print("xp:",xp)
        
        return xp,Pp




    def likelihood(self,parameters):
        

        self.Ai = parameters["Ai"]
        self.phase_i = parameters["phase_i"]

      
        
        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        self.Q = self.model.Q_function(self.gamma,self.sigma_p,self.dt)
        self.R = self.model.R_function(self.sigma_m,self.Npsr)
        self.F = self.model.F_function(self.gamma,self.dt)
        assert self.Q.shape ==self.F.shape
        assert self.R.shape == (self.Npsr,self.Npsr)
        

        #Initialise x and P
        x = np.zeros(2+self.Npsr)
        x[0] = parameters["phi0_gw"] #guess of the initial phase 
        x[1] = 1e-9 # initial guess of the omega
        x[2:2+self.Npsr] = self.observations[0,:]
        


                
        P = np.eye(len(x)) * self.sigma_p*1e10 #initial uncertainty on the states
        P[0,0] = 1e-10 #0.00 #uncertainty on initial phase
        P[1,1] = 1e-7 #uncertainty on omega_gw guess

        for i in range(self.Npsr):
            P[2+i,2+i] = 1e-1 #uncertainty on frequencies


        #Initialise the likelihood
        likelihood = 0.0
              

        x,P,l = self.update(x,P, self.observations[0,:])
        likelihood +=l

        #Place to store results
        x_results = np.zeros((self.Nsteps,len(x)))
        x_results[0,:] = x



        for i in np.arange(1,self.Nsteps):
            
            obs = self.observations[i,:]
           
            x_predict, P_predict   = self.predict(x,P,self.t[i])
            x,P,l = self.update(x_predict,P_predict, obs)
            likelihood +=l

            x_results[i,:] = x
            
        return likelihood,x_results

      






