


import numpy as np 
from gravitational_waves import GWs,gw_prefactor,gw_prefactor_optimised#,#gw_modulation_vectorized
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
        self.q_products = PTA.q_products.T
        self.t = PTA.t

        self.Npsr = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]


        print("The number of pulsars is:")
        print(self.Npsr)


        # #Initialise x and P
        self.x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        #self.P = np.eye(self.Npsr) * 1e5 #1e-13*1e9 
        self.P = np.ones(self.Npsr)*1e5

        self.identity = np.ones(self.Npsr)

    def get_likelihood(self,S,innovation):

        M = S.shape[1]
        x = np.linalg.solve(S,innovation)

        #print("likelihood")
        #print(np.linalg.slogdet(S)[-1], innovation, x, M*np.log(2*np.pi) )
        # -0.5*(slogdet(InnCov)[1] + Inn.T @ solve(InnCov, Inn)
        #            +  np.log(2*np.pi))
        return -0.5*(np.linalg.slogdet(S)[-1] + innovation @ x + M*np.log(2*np.pi))


    def get_vector_likelihood(self,S,innovation):
        x = innovation / S 
        #slogdet = np.log(np.prod(S)) 
        #return -0.5*(slogdet + innovation * x + self.Npsr*np.log(2*np.pi))
        #print(innovation)
        #print(x)
        return -0.5*(innovation @ x + self.Npsr*np.log(2*np.pi))









    # def log_likelihood(self,S,innovation):

    #     """
    #     Calculate the log likelihood for a given y,S
    #     """

    #     x=innovation
    #     cov=S
   
    #     flat_x = np.asarray(x).flatten()
    #     flat_mean = None


    #     l = multivariate_normal.logpdf(flat_x, flat_mean, cov)

    #     return l




    def update(self, x, P, observation,R,H):

      
       # H = self.model.H_function_i(modulation_factors)

    
        y = observation - H*x
        S = H*P*H + R
        K = P*H/S
        xnew = x + K*y
        I_KH = self.identity - K*H 
        Pnew = I_KH * P

        #print("Innovatin covariance is:")
        #print(S)
        
        l = self.get_vector_likelihood(S,y)
        #print(l)

        return xnew, Pnew,l


   # def predict(self,x,P,f,fdot,gamma,Q,t):

    def predict(self,x,P,F,T,Q):

       # F = self.model.F_function(gamma,self.dt)
        #T = self.model.T_function(f,fdot,gamma,t,self.dt)

       
        xp = F*x + T 
        Pp = F*P*F + Q

        
        return xp,Pp



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
        P = self.P

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
        
        #modulation_factors = gw_modulation_vectorized(self.t,parameters["omega_gw"],,prefactor,dot_product)
    
        #Precompute all the transition and control matrices
        F = self.model.F_function(gamma,self.dt)
        #print("gamma vector = ")
        #print(gamma)
        T = self.model.T_function(f,fdot,gamma,self.t,self.dt) #ntimes x npulsars
        

        #Initialise the likelihood
        likelihood = 0.0
        i = 0
        x,P,l= self.update(x,P, self.observations[i,:],R,modulation_factors[i,:])
        likelihood +=l

       #Place to store results
        #x_results = np.zeros((self.Nsteps,self.Npsr))
        #x_results[0,:] = x
        

        for i in np.arange(1,self.Nsteps):
            #print(i)
            x_predict, P_predict   = self.predict(x,P,F,T[i,:],Q)


            x,P,l = self.update(x_predict,P_predict, self.observations[i,:],R,modulation_factors[i,:])
            likelihood +=l

            #x_results[i,:] = x
            
        #return 1,2,3

        #print("finishing - get the l: ", likelihood)
        return likelihood#, x_results, P

      






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


#self.Npsr
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



