import arviz as az



import matplotlib.pyplot as plt 


idata = az.from_netcdf("filename.nc")


#az.plot_trace(idata, lines=[("m", {}, 5e-7)])
az.plot_trace(idata, lines=[("Omega", {}, 5e-7),("c", {}, 0.20)])

plt.show()