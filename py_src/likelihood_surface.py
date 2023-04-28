





import matplotlib.pyplot as plt 



import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import matplotlib.colors as mc
import numpy as np 
import sys 




p1 = sys.argv[1]
p2= sys.argv[2]
true_value_1= float(sys.argv[3])
true_value_2= float(sys.argv[4])



container = np.load(f"../data/{p1}_{p2}_surface_data_large_h_normalised.npz")


data_dict = {name: container[name] for name in container}

y = data_dict[p1]
x = data_dict[p2]
z = np.log10(np.abs(data_dict["likelihood"]))
z = data_dict["likelihood"]

print(np.max(z))
Y,X = np.meshgrid(y,x)
Z = z



print("Likelihood variance: ", np.var(z))
print("Likelihood max:", np.max(z))
print("Likelihood min:", np.min(z))
print("Likelihood max - min:", np.max(z) - np.min(z))


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z.T, cmap='viridis', alpha=0.6)
#ax.scatter(X, Y, Z.T, color='black', alpha=0.5, linewidths=1)
ax.set(xlabel=p1, ylabel=p2,zlabel="Log L")
#ax.set_zlabel('$f(y_1, y_2)$', labelpad=10)

ax.scatter(true_value_2,true_value_1,np.max(Z),c='r',s=100)

#ax.set_zlim(-4e18,1.0)

plt.show()