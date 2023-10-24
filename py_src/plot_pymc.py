import arviz as az



import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np 

import corner
from scipy.stats import gaussian_kde

mpl.rcParams.update({'font.size': 16})

def plotposts(samples, **kwargs):
    """
    Function to plot posteriors using corner.py and scipy's gaussian KDE function.
    """
    m=5e-7 
    c=0.20
    if "truths" not in kwargs:
        kwargs["truths"] = [m, c]

    fig = corner.corner(samples, labels=[r'$m$', r'$c$'], hist_kwargs={'density': True}, **kwargs)

    # plot KDE smoothed version of distributions
    for axidx, samps in zip([0, 3], samples.T):
        kde = gaussian_kde(samps)
        xvals = fig.axes[axidx].get_xlim()
        xvals = np.linspace(xvals[0], xvals[1], 100)
        fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')

    plt.show()





idata = az.from_netcdf("filename.nc")
mdata = idata.posterior.m # shape chains x num samples
cdata =  idata.posterior.c


#which chain to use
#could also just combine all the chains for one big posterior https://stats.stackexchange.com/questions/152037/combining-multiple-parallel-mcmc-chains-into-one-longer-chain
chain_idx = 0 
mdata = mdata[chain_idx,:]
cdata = cdata[chain_idx,:]



samples_pymc3_m = np.vstack((mdata.values,cdata.values)).T


resdict = {}
resdict['mpymc3_mu'] = np.mean(samples_pymc3_m[:,0])      # mean of m samples
resdict['mpymc3_sig'] = np.std(samples_pymc3_m[:,0])      # standard deviation of m samples
resdict['cpymc3_mu'] = np.mean(samples_pymc3_m[:,1])      # mean of m samples
resdict['cpymc3_sig'] = np.std(samples_pymc3_m[:,1])      # standard deviation of m samples
resdict['ccpymc3'] = np.corrcoef(samples_pymc3_m.T)[0,1]  # correlation coefficient between parameters

#get samples just from 1 chain

# samples_pymc3 = np.vstack((idata.posterior['m'], idata.posterior['c'])).T


# resdict['mpymc3_mu'] = np.mean(samples_pymc3[:,0])      # mean of m samples
# resdict['mpymc3_sig'] = np.std(samples_pymc3[:,0])      # standard deviation of m samples
# resdict['cpymc3_mu'] = np.mean(samples_pymc3[:,1])      # mean of c samples
# resdict['cpymc3_sig'] = np.std(samples_pymc3[:,1])      # standard deviation of c samples
# resdict['ccpymc3'] = np.corrcoef(samples_pymc3.T)[0,1]  # correlation coefficient between parameters

plotposts(samples_pymc3_m)

print(resdict)



#az.plot_trace(idata, lines=[("m", {}, 5e-7)])
#az.plot_trace(idata, lines=[("Omega", {}, 5e-7),("c", {}, 0.20)])

#plt.show()