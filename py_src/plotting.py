
import matplotlib.pyplot as plt 
from priors import priors_dict
import numpy as np 
import json
import pandas as pd 
import corner
import scienceplots
import sys 

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





def plot_all(t,states,measurements,predictions_x,predictions_y,psr_index,savefig=None):

    plt.style.use('science')

    tplot = t / (365*24*3600)
    state_i = states[:,psr_index]
    measurement_i = measurements[:,psr_index]
   



    h,w = 12,8
    rows = 4
    cols = 1
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)

    ax1.plot(tplot,state_i,label='state')
    ax2.plot(tplot,measurement_i,label="measurement",c="C3")


    try:


        prediction_i = predictions_x[:,psr_index]
        ax1.plot(tplot,prediction_i,label = 'prediction')




        prediction_i_y = predictions_y[:,psr_index]
        ax2.plot(tplot,prediction_i_y,label="prediction",c="C4")

        #Residuals
        residuals = prediction_i_y-measurement_i
        ax3.plot(tplot,residuals)

        print("Mean residual:", np.mean(residuals))
        ax4.hist(residuals,bins=50)

    except:
        print("Exception")
        pass 

    ax1.legend()
    ax2.legend()


    fs=18
    ax2.set_xlabel('t [years]', fontsize=fs)
    ax1.set_ylabel(r'$f_p$ [Hz]', fontsize=fs)
    ax2.set_ylabel(r'$f_M$ [Hz]', fontsize=fs)
    ax2.xaxis.set_tick_params(labelsize=fs-4)
    ax2.yaxis.set_tick_params(labelsize=fs-4)
    ax1.yaxis.set_tick_params(labelsize=fs-4)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.rcParams["font.family"] = "fantasy"

    if savefig != None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
   
    plt.show()



def plot_custom_corner(path,variables_to_plot,labels,injection_parameters,ranges,axes_scales,savefig):


    # Opening JSON file
    f = open(path)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    #Make it a dataframe. Nice for surfacing
    df = pd.DataFrame(data["samples"]["content"]) # posterior


    #Make it a dataframe. Nice for surfacing
    df_posterior = pd.DataFrame(data["posterior"]["content"]) # posterior
    df_samples = pd.DataFrame(data["samples"]["content"]) # posterior



    #Now make it a numpy array
    y_samp = df_samples.to_numpy() 


    print("Number of samples:")
    print(len(df_posterior))

    print("Medians/Variances")

    medians = df_posterior[variables_to_plot].median()
    variances = df_posterior[variables_to_plot].var()

    selected_variables = []
    selected_injections = []
    selected_labels = []
    for i in range(len(medians)):
        print(variables_to_plot[i], medians[i], variances[i])

        if variances[i] > 1e-40:
            selected_variables.extend([variables_to_plot[i]])
            selected_injections.extend([injection_parameters[i]])
            selected_labels.extend([labels[i]])
        else:
            print("Note! ", variables_to_plot[i], " has zero variance and will not be plotted")
   


    
    y_post = df_posterior[selected_variables].to_numpy()


    plt.style.use('science')

    try: 
        fig = corner.corner(y_post, 
                            color='C0',
                            show_titles=True,
                            smooth=True,smooth1d=True,
                            truth_color='C2',
                            quantiles=[0.16, 0.84],
                            truths = selected_injections,
                            range=ranges,
                            labels = selected_labels,
                            label_kwargs=dict(fontsize=16),
                            axes_scales = axes_scales)
            

        if savefig != None:
            plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
        
        
        plt.show()

    except:    # Get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()
        print(ex_type, ex_value)
        plt.close()


def iterate_over_priors(variable, variable_range,true_parameters,KF):

    
    guessed_parameters = true_parameters.copy()
    likelihoods=np.zeros_like(variable_range)

    
    i = 0
    for v in variable_range:
        
        guessed_parameters[variable] = v 
        model_likelihood,xres,yres = KF.likelihood(guessed_parameters)
        #print(model_likelihood)
        likelihoods[i] = model_likelihood
        #print(likelihoods[i] )
        i+=1 

    return likelihoods



def likelihoods_over_priors(parameters,priors,PTA,P,KF,sigma_p):



    plt.style.use('science')
    true_parameters = priors_dict(PTA,P)
    true_parameters["sigma_p"] = sigma_p
    #
    
    print("likelihoods_over_priors")

    h,w = 20,12
    rows = 6
    cols = 2
    fig, axes_object = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)
    


    axes = fig.get_axes()



    logvalues = ["omega_gw", "h"]

    i = 0
    for key,value in parameters.items():

        print(i, key, value)
        prior = priors[i]
        likelihood = iterate_over_priors(key, prior,true_parameters,KF)

        ax = axes[i]
        ax.plot(prior,likelihood)
        ax.set_xlabel(key, fontsize = 16)
        ax.axvline(value,linestyle='--', c='C2')

        #if key in logvalues:
           
           # ax.set_xscale('log')
        
        ax.set_yscale('log')

        #ax.set_yscale('log')
        i+=1


    

    plt.subplots_adjust(hspace=0.5)
    title = r"Likelihood Identifiability with $\sigma_p = $" + str(sigma_p)
    fig.suptitle(title, fontsize=20)

    print("setting linear xscale")
    ax.set_xscale("linear")
    plt.show()



def plot_likelihood(x,y):

    h,w = 20,12
    rows = 1
    cols = 1
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)


    z = [x for _, x in sorted(zip(y, x))]

    print(sorted(x))

    ax.plot(sorted(x),z)

    print("PLOTTING LIKELIHOOD")
    plt.show()





from scipy import interpolate

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
    if savefig != None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)













