
import matplotlib.pyplot as plt 
import numpy as np 
import json
import pandas as pd 
import corner
import scienceplots # noqa: F401
from scipy import interpolate

def plot_statespace(t,states,measurements,psr_index):


    tplot = t / (365*24*3600)
    state_i = states[:,psr_index]
    measurement_i = measurements[:,psr_index]

    print(len(tplot))
    print(state_i.shape)
    print(measurement_i.shape)

    h,w = 12,8
    rows = 2
    cols = 1
    fig, (ax1,ax2) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)

    ax1.plot(tplot,state_i)
    ax2.plot(tplot,measurement_i)
    plt.show()

def plot_all(t,states,measurements,measurements_clean,predictions_x,predictions_y,psr_index,savefig=None):

    plt.style.use('science')

    tplot = t / (365*24*3600)
    state_i = states[:,psr_index]
    measurement_i = measurements[:,psr_index]
    measurement_clean_i = measurements_clean[:,psr_index]



    h,w = 12,8
    rows = 4
    cols = 1
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)

    ax1.plot(tplot,state_i,label='state')

    #try:

    prediction_i = predictions_x[:,psr_index]
    ax1.plot(tplot,prediction_i,label = 'prediction')
    #except:
     #   print("Failed to plot the predictions for psr index ", psr_index)
    
    ax2.plot(tplot,measurement_i,label="measurement",c="C3")
    ax2.plot(tplot,measurement_clean_i,label="measurement_clean",c="C5")


   # try:
    prediction_i_y = predictions_y[:,psr_index]
    ax2.plot(tplot,prediction_i_y,label="prediction",c="C4")

    #Residuals
    residuals = prediction_i_y-measurement_i



    ax3.plot(tplot,residuals)

    print("Mean abs residual:", np.mean(np.abs(residuals)))
    ax4.hist(residuals,bins=50)

 


    ax1.legend()
    ax2.legend()


    fs=18
    ax2.set_xlabel('t [years]', fontsize=fs)
    ax1.set_ylabel(r'$f_p$ [Hz]', fontsize=fs)
    ax2.set_ylabel(r'$f_M$ [Hz]', fontsize=fs)
    ax3.set_ylabel(r'Residual [Hz]', fontsize=fs)
    ax2.xaxis.set_tick_params(labelsize=fs-4)
    ax2.yaxis.set_tick_params(labelsize=fs-4)
    ax1.yaxis.set_tick_params(labelsize=fs-4)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)


    if savefig is not None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
   

    plt.show()
    



def plot_custom_corner(path,variables_to_plot,labels,injection_parameters,ranges,axes_scales,scalings=[1.0,1.0],savefig=None,logscale=False,title=None,smooth=True,smooth1d=True):
    plt.style.use('science')

    if path.split('.')[-1] == 'json': #If it is a json files

        # Opening JSON file
        f = open(path)
        
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        print("The evidence is:", data["log_evidence"])
        f.close()
        #Make it a dataframe. Nice for surfacing
        df_posterior = pd.DataFrame(data["posterior"]["content"]) # posterior
    else: #is a parquet gzip
        df_posterior = pd.read_parquet(path) 

    

    #Make omega into nHz and also scale h
    df_posterior["omega_gw"] = df_posterior["omega_gw"]*scalings[0]
    df_posterior["h"] = df_posterior["h"]*scalings[1]
    injection_parameters[0] = injection_parameters[0] * scalings[0]
    injection_parameters[-1] = injection_parameters[-1] * scalings[1] 
    if ranges is not None: 
        ranges[0] = (ranges[0][0]*scalings[0],ranges[0][1]*scalings[0])
        ranges[-1] = (ranges[-1][0]*scalings[1],ranges[-1][1]*scalings[1])

    print("The number of samples is:", len(df_posterior))


    print("Variable/Injection/Median")
    medians = df_posterior[variables_to_plot].median()

    for i in range(len(medians)):
        print(variables_to_plot[i], injection_parameters[i], medians[i])
    print('-------------------------------')


    y_post = df_posterior[variables_to_plot].to_numpy()
    if logscale:
        y_post = np.log10(y_post)
        injection_parameters = np.log10(injection_parameters)
        ranges = np.log10(ranges)

    import warnings
    warnings.filterwarnings("error")

    
    #Now plot it using corner.corner
    fs = 20
    fig = corner.corner(y_post, 
                        color='C0',
                        show_titles=True,
                        smooth=smooth,smooth1d=smooth1d,
                        truth_color='C2',
                        quantiles=[0.16, 0.84], #[0.16, 0.84]
                        truths = injection_parameters,
                        range=ranges,
                        labels = labels,
                        label_kwargs=dict(fontsize=fs),
                        axes_scales = axes_scales)
            

    #Pretty-ify
    for ax in fig.axes:

        if ax.lines: #is anything plotted on this axis?
            
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            ax.yaxis.set_tick_params(labelsize=fs-6)
            ax.xaxis.set_tick_params(labelsize=fs-6)


        ax.title.set_size(18)
       


    if savefig is not None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
        
    if title is not None:
        fig.suptitle(title, fontsize=20)  
    plt.show()

    


def plot_likelihood(x,y,parameter_name,log_x_axes=False):

    h,w = 8,8
    rows = 1
    cols = 1
    fs =20
    plt.style.use('science')
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)


    ax.plot(x,y)


    ax.set_xlabel(parameter_name, fontsize=fs)
    ax.set_ylabel(r'$\log \mathcal{L}$', fontsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs-6)
    ax.xaxis.set_tick_params(labelsize=fs-6)

    if log_x_axes:
        ax.set_xscale('log')


    plt.show()







def SNR_plots(x,y1,y2,xlabel,savefig=None):

    plt.style.use('science')
   
    

    h,w = 12,8
    rows = 1
    cols = 1
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)
    

    ax.scatter(x,y1,label="Full PTA",c="C0")
    ax.scatter(x,y2,label="Single Pulsar",c="C2")

    

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(7,linestyle='--', c='0.5')

    f1 = interpolate.interp1d(y1, x)
    xc = f1(7.0)
    ax.axvline(xc,linestyle='--', c='C0')
    print("Cutoff value y1 = ", xc)
    idx = np.where(y1 > 7.0)[0]
    ax.plot(x[idx],y1[idx],c="C0")
    
    
    f2 = interpolate.interp1d(y2, x)
    xc = f2(7.0)
    ax.axvline(xc,linestyle='--', c='C2')
    print("Cutoff value y2 = ", xc)
    idx = np.where(y2 > 7.0)[0]
    ax.plot(x[idx],y2[idx],c="C2")




    fs=18
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$\Lambda$', fontsize=fs)
    
    ax.xaxis.set_tick_params(labelsize=fs-4)
    ax.yaxis.set_tick_params(labelsize=fs-4)

    ax.legend()
    if savefig is not None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)













