

import numpy as np
import matplotlib.pyplot as plt  




from matplotlib import ticker, cm
import matplotlib.pyplot as plt 

import scienceplots




import matplotlib

plt.style.use('science')
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)



path_to_earth_model = 'likelihood_surface_alpha_psi_mm_earth_coarse.npz'


def load_and_plot(path,ax):


    #Load the data
    data = np.load(path)
    alpha_values = data['x']
    psi_values = data['y']
    surface_pulsar = data['z'] 
    surface_pulsar = surface_pulsar / np.abs(np.max(surface_pulsar)) #Normalize

    #Extract location of maxima
    alpha_idx, psi_idx = np.unravel_index(surface_pulsar.argmax(), surface_pulsar.shape)
    xc = alpha_values[alpha_idx]
    yc = psi_values[psi_idx]
    zc = surface_pulsar[alpha_idx,psi_idx]

    print('Inferred maxima = ', xc,yc,zc)


    #Cast to 2D mesh
    X,Y = np.meshgrid(psi_values,alpha_values)
    lx = len(alpha_values)
    ly = len(psi_values)
    z = np.reshape(surface_pulsar, (lx, ly))



    #Plot colormap
    #CS = ax.pcolormesh(X, Y, z,clim=(2.0*np.max(surface_pulsar), np.max(surface_pulsar)),shading='gouraud',cmap='viridis')
    CS = ax.pcolormesh(X, Y, z,shading='gouraud',cmap='viridis')
    eps=2e-5
    #CS = ax.pcolormesh(X, Y, z,clim=(1.0-eps, 1.0),shading='gouraud',cmap='viridis')


    clb = plt.colorbar(CS)



    #Config
    ax.scatter(1.00,2.50, s=20,c='C3') #Location of true maxima
    ax.scatter(2.70,1.01, s=20,c='C4') #Location of settled maxima
    #ax.set_xscale('log')
    #ax.scatter(yc,xc, s=20,c='C3')
    #ax.scatter(1.01,2.70, s=20,c='C4')


    


    fs = 20
    ax.set_xlabel(r'$\alpha$', fontsize=fs)
    ax.set_ylabel(r'$\psi$', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs-4)
    ax.yaxis.set_tick_params(labelsize=fs-4)
    clb.ax.tick_params(labelsize=fs-4) 
    clb.ax.set_title(r'$\log \mathcal{L}$',fontsize=fs-4)
    savefig = 'likelihood_surface_alpha_psi'
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    #plt.setp(ax.get_yticklabels()[0], visible=False)   #no 0th label to prevent overlap  
    plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)





load_and_plot(path_to_earth_model,ax)
plt.show()