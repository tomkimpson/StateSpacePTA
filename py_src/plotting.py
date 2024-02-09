
import matplotlib.pyplot as plt 
import numpy as np 
import json
import pandas as pd 
import corner
import scienceplots # noqa: F401
from scipy import interpolate
import warnings
import random
from parse import * 
warnings.filterwarnings("error")
plt.style.use('science')
import scipy.stats as ss

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

    #prediction_i = predictions_x[:,psr_index]
    #ax1.plot(tplot,prediction_i,label = 'prediction')
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
    
def _extract_posterior_results(path,variables_to_plot,injection_parameters,ranges,scalings=[1.0,1.0]):

    print("Extracting data from file: ", path)

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
    if injection_parameters is not None:
        injection_parameters[0] = injection_parameters[0] * scalings[0]
        injection_parameters[-1] = injection_parameters[-1] * scalings[1] 
    if ranges is not None: 
        ranges[0] = (ranges[0][0]*scalings[0],ranges[0][1]*scalings[0])
        ranges[-1] = (ranges[-1][0]*scalings[1],ranges[-1][1]*scalings[1])

    print("The number of samples is:", len(df_posterior))


    print("Variable/Injection/Median")
    medians = df_posterior[variables_to_plot].median()

    if injection_parameters is not None:

        for i in range(len(medians)):
            print(variables_to_plot[i], injection_parameters[i], medians[i])

    else:
        for i in range(len(medians)):
            print(variables_to_plot[i], None, medians[i])
    print('-------------------------------')


    y_post = df_posterior[variables_to_plot].to_numpy()

    return_code = 0
    print("VARS TO PLOT ARE:", variables_to_plot)
    if "psi_gw" in variables_to_plot:
        if medians[2] < 2.0: #if the medians psi is weird, as sometimes happens, don't plot it 
            print("median psi is weird and won't match axis limits")
            return_code = 1

    if "phi0_gw" in variables_to_plot:
        if medians[1] < 0.08: #if the medians psi is weird, as sometimes happens, don't plot it 
            print("median phi0gw is weird and won't match axis limits")
            return_code = 1

    return y_post,injection_parameters,ranges,return_code



def plot_custom_corner(path,variables_to_plot,labels,injection_parameters,ranges,axes_scales,scalings=[1.0,1.0],savefig=None,logscale=False,title=None,smooth=True,smooth1d=True,fig=None):
    #Extract the data as a numpy array
    y_post,injection_parameters,ranges,return_code= _extract_posterior_results(path,variables_to_plot,injection_parameters,ranges,scalings=scalings)


    #Log scale the axes if needed
    if logscale:
        y_post = np.log10(y_post)
        injection_parameters = np.log10(injection_parameters)
        ranges = np.log10(ranges)



    print(ranges)
    #Now plot it using corner.corner
    fs = 20
    newfig = corner.corner(y_post, 
                        color='C0',
                        show_titles=True,
                        smooth=smooth,smooth1d=smooth1d,
                        truth_color='C2',
                        quantiles=[0.05, 0.95], #[0.16, 0.84]
                        truths = injection_parameters,
                        range=ranges,
                        labels = labels,
                        label_kwargs=dict(fontsize=fs),
                        axes_scales = axes_scales,
                        fig=fig)
            

    #Pretty-ify
    for ax in newfig.axes:

        if ax.lines: #is anything plotted on this axis?
            
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            ax.yaxis.set_tick_params(labelsize=fs-6)
            ax.xaxis.set_tick_params(labelsize=fs-6)


        ax.title.set_size(18)
       


    if savefig is not None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
        
    if title is not None:
        newfig.suptitle(title, fontsize=20)  
    #plt.show()

    









    

def _drop_braces(string_object):

    string_object = string_object.replace('{', '')
    string_object = string_object.replace('}', '')
    return string_object

def _extract_value_from_title(title_string):

    template = '$\\{param_name}$ = ${value}_{lower}^{upper}$'

    parsed_output = parse(template, title_string)

    if parsed_output is None: #Handles h which is not a greek letter
        template = '${param_name}$ = ${value}_{lower}^{upper}$'
        parsed_output = parse(template, title_string)


    return parsed_output['param_name'],_drop_braces(parsed_output['value']),_drop_braces(parsed_output['lower']),_drop_braces(parsed_output['upper'])

#https://stackoverflow.com/questions/32923605/is-there-a-way-to-get-the-index-of-the-median-in-python-in-one-command
def _argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]

def stacked_corner(list_of_files,number_of_files_to_plot,variables_to_plot,labels,injection_parameters,ranges,axes_scales,scalings=[1.0,1.0],savefig=None,logscale=False,title=None,smooth=True,smooth1d=True,seed=1,no_titles=False,plot_datapoints=True):

    #Some arrays to hold the title value returned by corner.corner
    num_params = len(variables_to_plot)
    title_values = np.zeros((num_params,number_of_files_to_plot)) #an array with shape number of parameters x number of noise realisations 
    title_upper = np.zeros((num_params,number_of_files_to_plot)) 
    title_lower = np.zeros((num_params,number_of_files_to_plot))


    #Select some files at random
    fig= None 
    random.seed(seed)
    selected_files = list_of_files
    #selected_files = random.sample(list_of_files,number_of_files_to_plot)
    #selected_files = list_of_files

    error_files = []
    for i,f in enumerate(selected_files):
        injection_parameters_idx = injection_parameters.copy()
        ranges_idx = ranges.copy()

        y_post,injection_parameters_idx,ranges_idx,return_code= _extract_posterior_results(f,variables_to_plot,injection_parameters_idx,ranges_idx,scalings=scalings)

        if return_code == 1:
            print("Breaking due to weird psi")
            continue

        errors = get_posterior_accuracy2(y_post,injection_parameters_idx,labels)
        error_files.extend([errors])
        k = i 
        if k ==2:
            k = k+1 #convoluted way of skipping C2 color. Surely a better way exists


        if logscale:
            yplot = np.log10(y_post)
            injection_parameters = np.log10(injection_parameters)
 
        else:
            yplot =y_post
        
        nsamples = len(y_post)
        fs = 20
   
        fig = corner.corner(yplot, 
                            color=f'C{k}',
                            show_titles=True,
                            smooth=smooth,smooth1d=smooth1d,
                            truth_color='C2',
                            quantiles=None, #[0.16, 0.84],
                            truths =injection_parameters_idx ,
                            range=ranges_idx,
                            labels = labels,
                            label_kwargs=dict(fontsize=fs),
                            axes_scales = axes_scales,
                            weights = np.ones(nsamples)/nsamples,
                            plot_datapoints=plot_datapoints,fig=fig)


        #Extract the axis titles 
        kk = 0
        for ax in fig.axes:
            ax_title = ax.get_title()
            
            if ax_title != '':

                print("debug:",kk,i)
                param_name, value,lower_limit,upper_limit = _extract_value_from_title(ax_title) #Get the values that corner.corner sends to the ax title
                title_values[kk,i] = value
                title_lower[kk,i] = lower_limit
                title_upper[kk,i] = upper_limit


            
                kk += 1
                


  


    #Pretty-ify
    ax_count = 0
    for ax in fig.axes:
        
        if ax.lines: #is anything plotted on this axis?            
            if len(ax.lines) == 18:
                ax_count += 1

            #ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            
            if ax_count == 5: #very hacky way to pop off overlapping ytick
                print("Setting y major locator")
                print("This is for axis:", ax)
                ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            else:
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            ax.yaxis.set_tick_params(labelsize=fs-6)
            ax.xaxis.set_tick_params(labelsize=fs-6)

            #Get all lines
            lines = ax.lines
            
            

        ax.title.set_size(18)



    #Get the indices of the median values from the list of medians 
    idxs = [] #this is the index of the median for each parameter. 
    for l in range(num_params):
        idx = _argmedian(title_values[l,:])
        idxs.extend([idx])



    #Now use it to set the titles
    kk = 0
    for ax in fig.axes:
        ax_title = ax.get_title()
        if ax_title != '':


            selected_idx = idxs[kk]



            new_title_string = rf'{labels[kk]} $= {title_values[kk,selected_idx]:.2f}_{{{title_lower[kk,selected_idx]:.2f}}}^{{+{title_upper[kk,selected_idx]:.2f}}}$'
            
            
            if no_titles:
                selected_idx=1
                new_title_string = rf'{labels[kk]} $= {title_values[kk,selected_idx]:.2f}_{{{title_lower[kk,selected_idx]:.2f}}}^{{+{title_upper[kk,selected_idx]:.2f}}}$'

                #ax.set_title('')
                ax.set_title(new_title_string, fontsize=18)
            else:
                ax.set_title(new_title_string, fontsize=18)
            
            kk += 1






    if savefig != None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)



    #Surface some numbers
    print("Surfacing numbers for comparing two posteriors")
    print("The RMSRE is:")

    if len(selected_files) ==2:
        earth_term_errors = error_files[0]
        pulsar_term_errors = error_files[1]

        #print(errors1)
        #print(errors2)
        #relative_error = (errors2 - errors1) / errors1
        difference = (earth_term_errors - pulsar_term_errors) #/ errors1




        percentage_difference =difference/ earth_term_errors

        #print(relative_error)
        for i in range(len(earth_term_errors)):
            print(variables_to_plot[i], "%.3g" % earth_term_errors[i],"%.3g" %pulsar_term_errors[i],"%.3g" %difference[i],"%.3g" %percentage_difference[i]) #printing to 3 sig fig



def get_posterior_accuracy(posterior,injection,labels):

    print("Gettti error in the 1D posteriors is as follows:")
    rmse_errors =np.zeros(posterior.shape[-1]) # vector of length n parameters
    for i in range(posterior.shape[-1]):
        y = posterior[:,i]
        inj = injection[i]
        
        #error = np.mean(np.abs(inj - y) / inj) #i.e. average absolute error

        rmse = np.sqrt(np.sum((y - inj)**2) / len(y)) # RMSRE https://stats.stackexchange.com/questions/413209/is-there-something-like-a-root-mean-square-relative-error-rmsre-or-what-is-t
        rmse_errors[i] = rmse

        #print(labels[i], error,rmse)
    #print('*****************************')


    return rmse_errors


#This one is used for submitted paper after discussion with AM
def get_posterior_accuracy2(posterior,injection,labels):

    print("Gettti error in the 1D posteriors is as follows:")
    rmse_errors =np.zeros(posterior.shape[-1]) # vector of length n parameters
    for i in range(posterior.shape[-1]):
        y = posterior[:,i]
        inj = injection[i]
        
        ymode = ss.mode(y.flatten(),keepdims=True)[0] #https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array?answertab=scoredesc#tab-top
        

        ymode = np.median(y)



        rmse = np.abs(ymode - inj) / inj
        
        
        rmse_errors[i] = rmse

        print(labels[i], ymode, inj,rmse)
    #print('*****************************')


    return rmse_errors




def plot_likelihood(x,y,parameter_name,log_x_axes=False,injection=1.0):

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


    ax.axvline(injection,c='0.5',linestyle='--')
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













