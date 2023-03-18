


import numpy as np 


class KalmanFilter:
    """
    A class to implement the Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

    """

    def __init__(self,Model, Observations,dt):

        """
        Initialize the class. 
        """

        self.model = Model
        self.observations = Observations
        self.dt = dt

        self.Npsr = self.observations.shape[-1]



    def update(self,x,P, observation,t,parameters,q,R):

    
        H = self.model.H_function(parameters,t,q)
    #     y = observation - H*x 
        
    #     S = H*P*H' .+ R 
    #     K = P*H'*inv(S)
    #     xnew = x .+ K*y

    

    #     #Update the covariance 
    #     #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    #     # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    #     # and works for non-optimal K vs the equation
    #     # P = (I-KH)P usually seen in the literature.
    #     I_KH = I - (K*H)
    #     KH = K*H
    #     #Pnew = I_KH * P * I_KH' .+ K * R * K'
    #     Pnew = I_KH * P

    # l = get_likelihood(S,y)


    #     return xnew, Pnew,l
    # end 




    def likelihood(self,parameters):

        Q = self.model.Q_function(parameters.gamma,parameters.sigma_p,self.dt)
        R = self.model.R_function(self.Npsr,parameters.sigma_m)


        x = self.observations[0,:] # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.eye(self.Npsr) * 1e-6*1e9 

        print(parameters)
        #x,P = update(x,P, observations[0,:],t[1],parameters,q,R)









