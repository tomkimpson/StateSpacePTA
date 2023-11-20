import sys
from run import bilby_inference_run,pp_plot_run

#Currently these are passed as command line arguments
#Convenient, but could be better to setup a json/config file to read from for reproducibility 
arg_name = sys.argv[1]           # reference name
h        =  float(sys.argv[2])   # strain
measurement_model =  sys.argv[3] # whether to use the H0(null) or H1(earth/pulsar) model
seed = int(sys.argv[4])          # the seeding


#Extra params for pp plot
omega =  float(sys.argv[5]) 
phi_0 =  float(sys.argv[6]) 
psi   =  float(sys.argv[7]) 
delta =  float(sys.argv[8]) 
alpha =  float(sys.argv[9]) 

if __name__=="__main__":
       pp_plot_run(arg_name,h,measurement_model,seed,omega,phi_0,psi,delta,alpha)





