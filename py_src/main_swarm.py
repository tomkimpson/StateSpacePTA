




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
#from plotting import plot_all
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict
from bilby_wrapper import BilbyLikelihood

import numpy as np
import bilby
import pyswarms as ps

if __name__=="__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")


    P   = SystemParameters()       #define the system parameters as a class
    PTA = Pulsars(P)               #setup the PTA
    data = SyntheticData(PTA,P) #generate some synthetic data
    

    #Define the model 
    model = LinearModel

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    # Run the KF once with the correct parameters.
    # This allows JIT precompile
    guessed_parameters = priors_dict(PTA,P)
    model_likelihood = KF.likelihood(guessed_parameters)
    print("Ideal likelihood = ", model_likelihood)

    def helper(y):
        #print("helper y = ", y)
        guessed_parameters = priors_dict(PTA,P)

        guessed_parameters["alpha_gw"] = y[0]
        guessed_parameters["delta_gw"] = y[1]
       
        return -KF.likelihood(guessed_parameters)

    def cost_function(x):

            n_particles = x.shape[0]  # number of particles


            dist = [helper(x[i,:]) for i in range(n_particles)]


            return np.array(dist)

            # print("This is the cost function")
            # print("The x value inputs is:")
            # print(len(x))
            # print(x)
            # l_returns = np.zeros_like(x)

            # for i in range(len(x)):
            #     print(i)
  
                
            #     print("Attemping l returns")
            #     l_returns[i] = KF.likelihood(guessed_parameters)


            # print(l_returns)
            # return l_returns




    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.3}

    # Create bounds
    max_bound =  np.pi * np.ones(2)
    min_bound = 0.0 * np.ones(2)
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=bounds)


    cost, pos = optimizer.optimize(cost_function, iters=1000)


    print(cost, pos)


