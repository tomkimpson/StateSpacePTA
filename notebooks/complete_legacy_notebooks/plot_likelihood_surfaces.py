

import numpy as np
import matplotlib.pyplot as plt  





import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(111)








path_to_earth_model = 'likelihood_surface_h_iota_mm_earth.npz'
path_to_earth_model = 'likelihood_surface_h_iota_mm_earth_coarse.npz'


# path1 = 'tmp_likelihood_surface.npz'
# path2 = 'tmp_likelihood_surface_earth.npz'

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
    ax.plot_surface(X, Y, z,alpha=0.5)

    #CS = ax.pcolormesh(X, Y, z) #clim=(0.0, 5e5)
    #plt.contour(X, Y, Z, 4, colors='k')
    #plt.colorbar(CS)
   # ax.set_xscale('log')



    print(path, yc,xc,zc)
    ax.scatter(yc,xc,zc, s=20)
    ax.scatter(1e-12,1.0,zc, s=20)


    # https://stackoverflow.com/questions/54495612/numpy-get-index-of-row-with-second-largest-value
    row,col = np.unravel_index(np.argsort(surface_pulsar.ravel()),surface_pulsar.shape)
    row,col = row[::-1],col[::-1]
    
    for k in np.arange(10):
        idx=k
        row_i,col_i = row[idx], col[idx]

        xc = iota_values[row_i]
        yc = h_values[col_i]
        zc = surface_pulsar[row_i,col_i]

        ax.scatter(yc,xc,zc, s=20,c='k')
        print(k,zc)

 

    fs = 20
    ax.set_xlabel(r'$h_0$', fontsize=fs)
    ax.set_ylabel(r'$\iota$', fontsize=fs)
    ax.set_zlabel(r'$ \log \mathcal{L}$')




#load_and_plot('tmp_likelihood_surface2_earth.npz',ax)
load_and_plot(path_to_earth_model,ax)
#load_and_plot(path2,ax)

#ax.scatter(1e-12,1.0,-586119.6803021401,c='r', s=20)
plt.show()