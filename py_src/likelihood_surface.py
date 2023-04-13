





import matplotlib.pyplot as plt 



import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import matplotlib.colors as mc
import numpy as np 

#container = np.load("../data/omega_delta_heatmap_data.npz")

container = np.load("../data/psi_gw_phi0_gw_surface_data.npz")




data_dict = {name: container[name] for name in container}

y = data_dict["phi0_gw"]
x = data_dict["psi_gw"]
z = np.log10(np.abs(data_dict["likelihood"]))
z = data_dict["likelihood"]

print(np.max(z))


Y,X = np.meshgrid(y,x)
Z = z






fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, Z.T, cmap='viridis', alpha=0.6)
ax.scatter(X, Y, Z.T, color='black', alpha=0.5, linewidths=1)
#ax.set(xlabel='$omega$', ylabel='$delta$')
#ax.set_zlabel('$f(y_1, y_2)$', labelpad=10)

ax.scatter(0.20,2.50,0,c='r',s=100)

plt.show()