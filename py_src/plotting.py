
import matplotlib.pyplot as plt 
from priors import priors_dict
import numpy as np 
import json
import pandas as pd 
import corner
import scienceplots

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





def plot_all(t,states,measurements,predictions,psr_index,savefig=None):

    plt.style.use('science')

    tplot = t / (365*24*3600)
    state_i = states[:,psr_index]
    measurement_i = measurements[:,psr_index]
    prediction_i = predictions[:,psr_index]


    h,w = 12,8
    rows = 2
    cols = 1
    fig, (ax1,ax2) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)

    ax1.plot(tplot,state_i,label='state')
    ax1.plot(tplot,prediction_i,label = 'prediction')
    ax2.plot(tplot,measurement_i,c='C2')


    ax1.legend()

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



def plot_custom_corner(path,labels, injection_parameters,axes_scales):


    # Opening JSON file
    f = open(path)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    #Make it a dataframe. Nice for surfacing
    df = pd.DataFrame(data["samples"]["content"]) # posterior
    y = df.to_numpy() 


    plt.style.use('science')

    
    fig = corner.corner(y,
                        color='C0',
                        show_titles=True,
                        smooth=True, 
                        smooth1d=True,
                        labels=labels,
                        truth_color='C2',
                        quantiles=[0.16, 0.84],
                        truths=injection_parameters,
                        axes_scales = axes_scales)
    
    # fig.set_figwidth(12)
    # fig.set_figheight(8)

    # axes = fig.get_axes()
    # axes[0].set_xlim(-1e-9,1e-5)
    # axes[0].set_xscale('log')

       
    

    plt.show()


def iterate_over_priors(variable, variable_range,true_parameters,KF):

    
    guessed_parameters = true_parameters.copy()
    likelihoods=np.zeros_like(variable_range)

    
    i = 0
    for v in variable_range:
        
        guessed_parameters[variable] = v 
        model_likelihood,model_state_predictions = KF.likelihood_and_states(guessed_parameters)
        likelihoods[i] = np.abs(model_likelihood)
        i+=1 

    return likelihoods



def likelihoods_over_priors(parameters,priors,PTA,P,KF,sigma_p):



    plt.style.use('science')
    true_parameters = priors_dict(PTA,P)
    true_parameters["sigma_p"] = sigma_p
    

    h,w = 20,12
    rows = 5
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

        if key in logvalues:
           
            ax.set_xscale('log')
            #ax.set_yscale('log')

        ax.set_yscale('log')
        i+=1


    

    plt.subplots_adjust(hspace=0.5)
    title = r"Likelihood Identifiability with $\sigma_p = $" + str(sigma_p)
    fig.suptitle(title, fontsize=20)

   
    plt.show()






from scipy import interpolate

def SNR_plots(x,y,xlabel):

    plt.style.use('science')
   
    

    h,w = 20,12
    rows = 1
    cols = 1
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)
    

    ax.scatter(x,y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(7,linestyle='--', c='0.5')

    f = interpolate.interp1d(y, x)

    xc = f(7.0)
    ax.axvline(xc,linestyle='--', c='0.5')
    print("Cutoff value = ", xc)

    

    fs=18
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$\Lambda$', fontsize=fs)
    
    ax.xaxis.set_tick_params(labelsize=fs-4)
    ax.yaxis.set_tick_params(labelsize=fs-4)













