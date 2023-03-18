
import matplotlib.pyplot as plt 

def plot_simple(t,states,measurements,psr_index):


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