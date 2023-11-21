import sys
from run import bilby_inference_run

#Currently these are passed as command line arguments
#Convenient, but could be better to setup a json/config file to read from for reproducibility 
arg_name = sys.argv[1]           # reference name
h        =  float(sys.argv[2])   # strain
measurement_model =  sys.argv[3] # whether to use the H0(null) or H1(earth/pulsar) model
seed = int(sys.argv[4])          # the seeding


if __name__=="__main__":
    bilby_inference_run(arg_name,h,measurement_model,seed)






