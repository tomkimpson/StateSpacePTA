


import jax.numpy as np
from gravitational_waves import gw_prefactor_optimised
from jax import jit


"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
@jit
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)
    return -0.5*(np.dot(innovation,x) + N*np.log(2*np.pi))


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
@jit
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
@jit
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


    
    #@jit
    def likelihood_test(self,omega_gw,
                        phi0_gw,
                        psi_gw,
                        iota_gw,
                        delta_gw,
                        alpha_gw,
                        h,
                        f,
                        fdot,
                        gamma, 
                        d,
                        sigma_p,
                        sigma_m
                   ):
        


        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        Q = self.model.Q_function(gamma,sigma_p,self.dt)
        R = self.model.R_function(sigma_m)
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars


        #Initialise x and P
        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr) * sigma_m*1e10 


        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
        modulation_factors = gw_prefactor_optimised(delta_gw,
                                                    alpha_gw,
                                                    psi_gw,
                                                    self.q,
                                                    self.q_products,
                                                    h,
                                                    iota_gw,
                                                    omega_gw,
                                                    d,
                                                    self.t,
                                                    phi0_gw
                                                    )


        #Initialise the likelihood
        likelihood = 0.0
              

        #Perform the first update step
        x,P,l = update(x,P, self.observations[0,:],R,modulation_factors[0,:])
        likelihood +=l

        for i in np.arange(1,self.Nsteps):
            obs = self.observations[i,:]
           
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            #x,P,l = update(x_predict,P_predict, obs,R,modulation_factors[1,:])
            x,b,l = update(x_predict,P_predict, obs,R,modulation_factors[1,:])


            likelihood += l #1.0*x[0]*modulation_factors[0,0]




        #return np.sum(self.observations) * omega_gw
        return likelihood #* omega_gw






    def likelihood(self,omega_gw,
                        phi0_gw,
                        psi_gw,
                        iota_gw,
                        delta_gw,
                        alpha_gw,
                        h,
                        f,
                        fdot,
                        gamma, 
                        d,
                        sigma_p,
                        sigma_m
                   ):


        #print("This is the likelihood function call------------------------")
        
  
        #print("Precomputing Q,R,F,T------------------------")

        
        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        Q = self.model.Q_function(gamma,sigma_p,self.dt)
        R = self.model.R_function(sigma_m)
        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars


        #Initialise x and P
        #print("Initialise x and P------------------------")

        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr) * sigma_m*1e10 

       
     

        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
        #print("Precompute GW factor------------------------")

        modulation_factors = gw_prefactor_optimised(delta_gw,
                                                    alpha_gw,
                                                    psi_gw,
                                                    self.q,
                                                    self.q_products,
                                                    h,
                                                    iota_gw,
                                                    omega_gw,
                                                    d,
                                                    self.t,
                                                    phi0_gw
                                                    )

        #Initialise the likelihood
        likelihood = 0.0
              

        #print("The first GW step------------------------")

        x,P,l = update(x,P, self.observations[0,:],R,modulation_factors[0,:])
        likelihood +=l


        #print("Iterating over all observations------------------------")

        j_counter = 1
        for i in np.arange(1,self.Nsteps):
            #print("Iterating over all observations------------------------", i,j_counter)
            j_counter += 1
            
            obs = self.observations[i,:]
           
            x_predict, P_predict   = predict(x,P,F,T[i,:],Q)
            x,P,l = update(x_predict,P_predict, obs,R,modulation_factors[i,:])
            likelihood +=l
            #print("likelihood value = ", likelihood)

  
            

        #print("Iteration complete, returning net likelihood = ", likelihood)
        return likelihood  

      



