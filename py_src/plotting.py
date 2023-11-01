
import matplotlib.pyplot as plt 
import scienceplots # noqa: F401

# import numpy as np 
# import json
# import pandas as pd 
# import corner

# from scipy import interpolate
# import warnings
# import random
# from parse import * 



# warnings.filterwarnings("error")
plt.style.use('science')








#data is the synthetic data
#predictions is a list of [x_predictions,y_predictions]
def global_plot(data,predictions,psr_index=1):
    plt.style.use('science')


    #Extract the data
    tplot = data.t / (365*24*3600)
    state_i                 = data.intrinsic_frequency[:,psr_index]
    measurement_i           = data.f_measured[:,psr_index]
    predicted_state_i       = predictions[0][:,psr_index]
    predicted_measurement_i = predictions[1][:,psr_index]

    #Setup the figure
    rows = 3
    cols = 2
    h,w = 12,8
    plt.figure(figsize=(h,w))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (1, 0),sharex=ax1)
    ax3 = plt.subplot2grid((3, 2), (2, 0),sharex=ax1)
    ax4 = plt.subplot2grid((3, 2), (0, 1),rowspan=3)





    # rows = 4
    # cols = 1
    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)




    #In the top panel plot the state and the state prediciton
    ax1.plot(tplot,state_i,label='state')
    ax1.plot(tplot,predicted_state_i,label='predicted state')

    #In the 2nd panel plot the measurement and the measurement prediction
    ax2.plot(tplot,measurement_i,label='measurement')
    ax2.plot(tplot,predicted_measurement_i,label='predicted measurement')

    #In the 3rd panel plot the difference in measurment space between predicted and actual
    residuals = predicted_measurement_i-measurement_i
    ax3.plot(tplot,residuals,c='C3')

    # In the bottom panel plot a histogram of the residuals - these should be gaussian
    ax4.hist(residuals,bins=50,color = 'C4')
    

    #Formatting

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

    plt.subplots_adjust(wspace=0.1, hspace=0.0)

    plt.show()

  


def plot_all(t,states,measurements,measurements_clean,predictions_x,predictions_y,psr_index,savefig=None):

  




  

 



    if savefig is not None:
        plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
   

    plt.show()
   