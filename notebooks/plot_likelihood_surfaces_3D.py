

import numpy as np
import matplotlib.pyplot as plt  




from matplotlib import ticker, cm
import matplotlib.pyplot as plt 

import scienceplots




import matplotlib

plt.style.use('science')
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')




path_to_earth_model = 'likelihood_surface_h_iota_mm_earth.npz'
path_to_earth_model = 'likelihood_surface_h_iota_mm_earth_broader.npz'
path_to_earth_model = 'likelihood_surface_h_iota_mm_earth_coarse.npz'

path_to_earth_model = 'likelihood_surface_h_iota_mm_earth_coarse_small_h.npz'
def load_and_plot(path,ax):


    #Load the data
    data = np.load(path)
    iota_values = data['x']
    h_values = np.log10(data['y'])
    surface_pulsar = data['z'] 
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
    ax.set_xlabel(r'$h_0$', fontsize=fs)
    ax.set_ylabel(r'$\iota$', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs-4)
    ax.yaxis.set_tick_params(labelsize=fs-4)
    
load_and_plot(path_to_earth_model,ax)
plt.show()