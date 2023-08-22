

import numpy as np
import matplotlib.pyplot as plt  




from matplotlib import ticker, cm
import matplotlib.pyplot as plt 

import scienceplots




import matplotlib

plt.style.use('science')
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)




path_to_earth_model = 'likelihood_surface_h_iota_mm_earth.npz'
path_to_earth_model = 'likelihood_surface_h_iota_mm_earth_broader.npz'
path_to_earth_model = 'likelihood_surface_h_iota_mm_earth_coarse.npz'


def load_and_plot(path,ax):

    data = np.load(path)


    iota_values = data['x']
    h_values = data['y']
    #surface_pulsar = np.log(np.abs(data['z']))
    surface_pulsar = data['z'] 
    surface_pulsar = surface_pulsar / np.abs(np.max(surface_pulsar))

    iota_idx, h_idx = np.unravel_index(surface_pulsar.argmax(), surface_pulsar.shape)




    print(np.min(surface_pulsar),np.max(surface_pulsar))


    xc = iota_values[iota_idx]
    yc = h_values[h_idx]
    zc = surface_pulsar[iota_idx,h_idx]


    X,Y = np.meshgrid(h_values,iota_values)


   
    lx = len(iota_values)
    ly = len(h_values)
    z = np.reshape(surface_pulsar, (lx, ly))



    #Colormap
    #)
    CS = ax.pcolormesh(X, Y, z,clim=(2.0*np.max(surface_pulsar), np.max(surface_pulsar)),shading='gouraud',cmap='viridis')
    clb = plt.colorbar(CS)




    #Contour f
    #plt.contour(X, Y, z, 50,vmin = 2.0*np.max(surface_pulsar), vmax = np.max(surface_pulsar), cmap=cm.coolwarm) #cmap=cm.PuBu_r



    



    ax.set_xscale('log')




    #print(path, yc,xc,zc)
    ax.scatter(yc,xc, s=20,c='C3')
    #ax.scatter(1e-12,1.0,zc, s=20)

 

    fs = 20
    ax.set_xlabel(r'$h_0$', fontsize=fs)
    ax.set_ylabel(r'$\iota$', fontsize=fs)
    ax.xaxis.set_tick_params(labelsize=fs-4)
    ax.yaxis.set_tick_params(labelsize=fs-4)
    clb.ax.tick_params(labelsize=fs-4) 
    clb.ax.set_title(r'$\log \mathcal{L}$',fontsize=fs-4)
    savefig = 'likelihood_surface'
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.setp(ax.get_yticklabels()[0], visible=False)   #no 0th label to prevent overlap  


    plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)





load_and_plot(path_to_earth_model,ax)

plt.show()