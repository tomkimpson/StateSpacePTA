

import numpy as np
import matplotlib.pyplot as plt  





import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(6,6))
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)











path1 = 'tmp_likelihood_surface.npz'
path2 = 'tmp_likelihood_surface_earth.npz'

def load_and_plot(path,ax):

    data = np.load(path)


    iota_values = data['x']
    h_values = data['y']
    surface_pulsar = data['z']
    iota_idx, h_idx = np.unravel_index(surface_pulsar.argmax(), surface_pulsar.shape)



    xc = iota_values[iota_idx]
    yc = h_values[h_idx]
    zc = surface_pulsar[iota_idx,h_idx]


    X,Y = np.meshgrid(h_values,iota_values)


   
    lx = len(iota_values)
    ly = len(h_values)
    z = np.reshape(surface_pulsar, (lx, ly))
   
    # Plot a 3D surface
    #ax.plot_surface(X, Y, z,alpha=0.5)

    CS = ax.pcolormesh(X, Y, z) #clim=(0.0, 5e5)
    #plt.contour(X, Y, Z, 4, colors='k')
    plt.colorbar(CS)
    ax.set_xscale('log')



    print(path, yc,xc,zc)
    #ax.scatter(yc,xc,zc, s=20)
 



#load_and_plot('tmp_likelihood_surface2_earth.npz',ax)
load_and_plot(path1,ax)
#load_and_plot(path2,ax)

#ax.scatter(1e-12,1.0,-586119.6803021401,c='r', s=20)
plt.show()