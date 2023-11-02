
import matplotlib


import sys
import json 
import pandas as pd 
import numpy as np 
try:
    sys.path.remove("../py_src") # Hacky way to add higher directory to python modules path. 
except:
    pass
sys.path.append("../py_src") # Means that I dont have to make src/ a proper python package



from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import bilby_priors_dict
import logging 
import numpy as np 
import bilby







h = 1 #doesnt matter
measurement_model = 'pulsar' 
seed = 1237
num_gw_sources = 2 


#Setup the system
P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,num_gw_sources=num_gw_sources) # define the system parameters as a dict. Todo: make this a class
PTA = Pulsars(P)                                       # setup the PTA
data = SyntheticData(PTA,P)                            # generate some synthetic data

#Define the model 
model = LinearModel(P)

#Initialise the Kalman filter
KF = KalmanFilter(model,data.f_measured,PTA)



init_parameters_optimal, priors_optimal = bilby_priors_dict(PTA,P,set_parameters_as_known=True)
optimal_parameters = priors_optimal.sample(1)
model_likelihood = KF.likelihood(optimal_parameters)

import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')





def generate_likelihood_surface(xx,yy,parameter):


    likelihood_surface= np.zeros((len(xx), len(yy)))


    for i in range(len(xx)):
        for j in range(len(yy)):
            suboptimal_parameters = optimal_parameters.copy()
            suboptimal_parameters[f'{parameter}_0'] = np.array(xx[i])
            suboptimal_parameters[f'{parameter}_1'] = np.array(yy[j])
            likelihood_surface[i][j] = KF.likelihood(suboptimal_parameters)
            #likelihood_surface[i][j],_,_ = KF.likelihood_with_results(suboptimal_parameters)


    return likelihood_surface

xx = np.arange(0,3,0.05)
yy = np.arange(0,3,0.05)



parameter = 'alpha_gw'
injections = P.α

phi0_ll = generate_likelihood_surface(xx,yy,parameter)







def load_and_plot(xx,yy,zz,ax):


    #Load the data
    
    iota_values = xx
    h_values = yy
    surface_pulsar = zz

    surface_pulsar = surface_pulsar / np.abs(np.max(surface_pulsar)) #Normalize


    double_log = False #Do you want to log the likelihood again?
    if double_log:
        surface_pulsar = np.log10(np.abs(surface_pulsar))
        #Extract location of minima
        iota_idx, h_idx = np.unravel_index(surface_pulsar.argmin(), surface_pulsar.shape)
    else:
        iota_idx, h_idx = np.unravel_index(surface_pulsar.argmax(), surface_pulsar.shape)

    
    
    xc = iota_values[iota_idx]
    yc = h_values[h_idx]
    zc = surface_pulsar[iota_idx,h_idx]


    #Cast to 2D mesh
    X,Y = np.meshgrid(h_values,iota_values)
    lx = len(iota_values)
    ly = len(h_values)
    z = np.reshape(surface_pulsar, (lx, ly))


    #Plot colormap
    ax.plot_surface(X, Y, z,alpha=0.5)
   

    #Config
  
    ax.scatter(yc,xc,zc, s=20,c='C3')
    fs = 20
    # ax.set_xlabel(r'$h_0$', fontsize=fs)
    # ax.set_ylabel(r'$\iota$', fontsize=fs)

    ax.set_xlabel(r'$\alpha$', fontsize=fs)
    ax.set_ylabel(r'$\psi$', fontsize=fs)

    # ax.xaxis.set_tick_params(labelsize=fs-4)
    # ax.yaxis.set_tick_params(labelsize=fs-4)

    #eps = 1e-3
    #ax.set_zlim(1.0-eps,1.0)

    plt.show()
    
load_and_plot(xx,yy,phi0_ll,ax)
















