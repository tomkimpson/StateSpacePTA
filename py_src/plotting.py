
import matplotlib.pyplot as plt 

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





def plot_all(t,states,measurements,predictions,psr_index):


    tplot = t / (365*24*3600)
    state_i = states[:,psr_index]
    measurement_i = measurements[:,psr_index]
    prediction_i = predictions[:,psr_index]
    print("Final prediction: ", prediction_i[-1])
    print("Final measurement: ", measurement_i[-1])


    h,w = 12,8
    rows = 2
    cols = 1
    fig, (ax1,ax2) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)

    ax1.plot(tplot,state_i)
    ax1.plot(tplot,prediction_i)
    ax2.plot(tplot,measurement_i)
    plt.show()





def plot_likelihood(x,y):


    h,w = 10,6
    rows = 1
    cols = 1
    fig, ax1 = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)

    ax1.scatter(x,abs(y))
    ax1.plot(x,abs(y))

    ax1.set_xscale('log')
    #ax1.set_yscale('log')
    ax1.axvline(1e-7,linestyle='--', c = '0.5')

    ax1.set_xlabel(r'$\omega$ [years]', fontsize=20)
    ax1.set_ylabel(r'$\mathcal{L}$', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)

    ax1.set_yscale("log")
    plt.show()