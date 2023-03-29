


import numpy as np 
from gravitational_waves import gw_prefactor_optimised
from scipy.stats import multivariate_normal

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


    def get_likelihood(self,S,innovation):

        M = S.shape[1]
        x = np.linalg.solve(S,innovation)

        #print("likelihood")
        #print(np.linalg.slogdet(S)[-1], innovation, x, M*np.log(2*np.pi) )
        # -0.5*(slogdet(InnCov)[1] + Inn.T @ solve(InnCov, Inn)
        #            +  np.log(2*np.pi))
        return -0.5*(np.linalg.slogdet(S)[-1] + innovation @ x + M*np.log(2*np.pi))



    def log_likelihood(self,S,innovation):

        """
        Calculate the log likelihood for a given y,S
        """

        x=innovation
        cov=S
   
        flat_x = np.asarray(x).flatten()
        flat_mean = None


        l = multivariate_normal.logpdf(flat_x, flat_mean, cov)

        return l




    def update(self, x, P, observation,t,parameters,R,modulation_factors):

        H = self.model.H_function(modulation_factors)

        y = observation - np.dot(H,x) 
        
        
        S = H@P@H.T + R 
        
        K = P@H.T@np.linalg.inv(S)
        xnew = x + K@y

    
    #     #Update the covariance 
    #     #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    #     # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    #     # and works for non-optimal K vs the equation
    #     # P = (I-KH)P usually seen in the literature.
        I_KH = np.eye(self.Npsr) - (K@H)
        Pnew = I_KH @ P @I_KH.T + K @ R @ K.T
        
        #l = self.get_likelihood(S,y)
        l = self.log_likelihood(S,y)


        #print("Covariance = ", S)
       
        return xnew, Pnew,l


    def predict(self,x,P,f,fdot,gamma,Q,t):


        F = self.model.F_function(gamma,self.dt)
        T = self.model.T_function(f,fdot,gamma,t,self.dt)

       
        xp = F@x + T 
        Pp = F@P@F.T + Q

        


        return xp,Pp



    def likelihood(self,parameters):
        
        f,fdot,gamma,d = map_dicts_to_vector(parameters)
        
        #Setup Q and R matrices.
        #These are time-independent functions of the parameters
        Q = self.model.Q_function(gamma,parameters["sigma_p"],self.dt)
        R = self.model.R_function(self.Npsr,parameters["sigma_m"])



        #Initialise x and P
        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.eye(self.Npsr) * parameters["sigma_m"]*1e10 #1e-13*1e9 

        #GW quantities

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
              

        x,P,l = self.update(x,P, self.observations[0,:], self.t[0],parameters,R,modulation_factors[0,:])
        
        likelihood +=l

        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr))
        x_results[0,:] = x





        for i in np.arange(1,self.Nsteps):
            
            obs = self.observations[i,:]
            ti = self.t[i]

            x_predict, P_predict   = self.predict(x,P,f,fdot,gamma,Q,ti)
            x,P,l = self.update(x_predict,P_predict, obs,ti,parameters,R,modulation_factors[i,:])
            likelihood +=l

            x_results[i,:] = x
            
        return likelihood, x_results, P

      




"""
Useful function which maps repeated quantities in the dictionary to a
vector
"""
def map_dicts_to_vector(parameters_dict):


    f = np.array([val for key, val in parameters_dict.items() if "f0" in key])
    fdot = np.array([val for key, val in parameters_dict.items() if "fdot" in key])
    gamma = np.array([val for key, val in parameters_dict.items() if "gamma" in key])
    d = np.array([val for key, val in parameters_dict.items() if "distance" in key])

    return f,fdot,gamma,d




