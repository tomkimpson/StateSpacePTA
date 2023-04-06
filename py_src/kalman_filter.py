


import numpy as np 
from gravitational_waves import gw_prefactor_optimised

from numba import jit 



"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
@jit(nopython=True)
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)
    #slogdet = np.sum(np.log(S)) # Uses log rules and diagonality of covariance "matrix"
    #eturn -0.5*(slogdet+innovation @ x + N*np.log(2*np.pi))
    return -0.5*(innovation @ x + N*np.log(2*np.pi))


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
@jit(nopython=True)
def update(x, P, observation,R,H):

    
    y    = observation - H*x
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


    






    def likelihood(self,parameters):
        
        #Bilby takes a dict
        #For us this is annoying - map some quantities to be vectors
        f,fdot,gamma,d = map_dicts_to_vector(parameters)


      
        
        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        Q = self.model.Q_function(gamma,parameters["sigma_p"],self.dt)
        R = self.model.R_function(parameters["sigma_m"])
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars


        #Initialise x and P
        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr) * parameters["sigma_m"]*1e10 

       
     

        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
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

        #Initialise the likelihood
        likelihood = 0.0
              

        x,P,l = update(x,P, self.observations[0,:],R,modulation_factors[0,:])
        likelihood +=l

        #Place to store results
        #x_results = np.zeros((self.Nsteps,self.Npsr))
        #x_results[0,:] = x





        for i in np.arange(1,self.Nsteps):
            
            obs = self.observations[i,:]
           
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, obs,R,modulation_factors[i,:])
            likelihood +=l

            #x_results[i,:] = x
            
        return likelihood  #x_results

      


    def likelihood_and_states(self,parameters):
        
        #Bilby takes a dict
        #For us this is annoying - map some quantities to be vectors
        f,fdot,gamma,d = map_dicts_to_vector(parameters)
        

        # for key,value in parameters.items():
        #     print(key, type(value))
        #print(parameters)

        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        Q = self.model.Q_function(gamma,parameters["sigma_p"],self.dt)
        R = self.model.R_function(parameters["sigma_m"])
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars


    
        #Initialise x and P
        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr,dtype=self.NF) * parameters["sigma_m"]*1e10 

    



        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
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
        

        

        #Initialise the likelihood
        likelihood = self.NF(0.0)
              

        x,P,l = update(x,P, self.observations[0,:],R,modulation_factors[0,:])
        likelihood +=l

  
        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr),dtype=self.NF)
        x_results[0,:] = x





        for i in np.arange(1,self.Nsteps):
        #for i in np.arange(1,3):
            
            obs = self.observations[i,:]
           
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, obs,R,modulation_factors[i,:])
            likelihood +=l

            x_results[i,:] = x
            
        return likelihood, x_results




    def null_likelihood_and_states(self,parameters):
        
        #Bilby takes a dict
        #For us this is annoying - map some quantities to be vectors
        f,fdot,gamma,d = map_dicts_to_vector(parameters)
        

    

        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        Q = self.model.Q_function(gamma,parameters["sigma_p"],self.dt)
        R = self.model.R_function(parameters["sigma_m"])
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars


        


        #Initialise x and P
        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr,dtype=self.NF) * parameters["sigma_m"]*1e10 

      
        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
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
        
        null_modulation_factors = np.ones_like(modulation_factors,dtype=self.NF)
        

        #Initialise the likelihood
        likelihood = self.NF(0.0)
              

        x,P,l = update(x,P, self.observations[0,:],R,null_modulation_factors[0,:])
        likelihood +=l

      

        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr),dtype=self.NF)
        x_results[0,:] = x





        for i in np.arange(1,self.Nsteps):
            
            obs = self.observations[i,:]
           
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, obs,R,null_modulation_factors[i,:])
            likelihood +=l

            x_results[i,:] = x
            
        return likelihood, x_results







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




