


import numpy as np 
from gravitational_waves import GWs,gw_prefactor
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

        self.Npsr = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]


        print("The number of pulsars is:")
        print(self.Npsr)


    def get_likelihood(self,S,innovation):

        M = S.shape[1]
        x = np.linalg.solve(S,innovation)

        #print("likelihood")
        #print(np.linalg.slogdet(S)[-1], innovation, x, M*np.log(2*np.pi) )
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




    def update(self,x,P, observation,t,parameters,R,prefactor,dot_product):

        H = self.model.H_function(t,parameters["omega_gw"],parameters["phi0_gw"],prefactor,dot_product)

        y = observation - np.dot(H,x) 
        
        
        S = H@P@H.T + R #H is diagonal, transpose is a waste. Kept in for generality

        
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
        P = np.eye(self.Npsr) * 1e9 #1e-13*1e9 

        #GW quantities
        gw = GWs(parameters)
        prefactor, dot_product =gw_prefactor(gw.n,self.q, gw.Hij, gw.omega_gw,d)
       

        #Initialise the likelihood
        likelihood = 0.0
       

        # print("Welcome to the linear Kalman filter in Julia")
        # print("You have chosen the following settings:")

        # print("f0")
        # print(f)

        # print("f0̇")
        # print(fdot)

        # print("gamma")
        # print(gamma)

        # print("d")
        # print(d)

        # print("Φ0")
        # print(parameters["phi0_gw"])

        # print("ψ")
        # print(parameters["psi_gw"])

        # print("ι")
        # print(parameters["iota_gw"])

        # print("δ")
        # print(parameters["delta_gw"])

        # print("α")
        # print(parameters["alpha_gw"])

        # print("h")
        # print(parameters["h"])

        # print("σp")
        # print(parameters["sigma_p"])
    
        # print("σm")
        # print(parameters["sigma_m"])
        
        # print("ω")
        # print(parameters["omega_gw"])





















        #The first update step
       

        x,P,l = self.update(x,P, self.observations[0,:],self.t[0],parameters,R,prefactor,dot_product)
        
        likelihood +=l

        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr))
        x_results[0,:] = x





        for i in np.arange(1,self.Nsteps):
        #for i in np.arange(1,5):

            
            obs = self.observations[i,:]
            ti = self.t[i]

            x_predict,P_predict   = self.predict(x,P,f,fdot,gamma,Q,ti)
            x,P,l = self.update(x_predict,P_predict, obs,ti,parameters,R,prefactor,dot_product)
            likelihood +=l

            x_results[i,:] = x


        

        return likelihood, x_results,P

      






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




