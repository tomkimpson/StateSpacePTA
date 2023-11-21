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
















def pp_plot_run(arg_name,h,measurement_model,seed,omega,phi_0,psi,delta,alpha):

    logger = logging.getLogger().setLevel(logging.INFO)
    #Setup the system
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed,Ω=omega,Φ0=phi_0,ψ=psi,α=alpha,δ=delta) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    optimal_parameters = priors_dict(PTA,P)
    model_likelihood = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")
    
    #Bilby
    init_parameters, priors = bilby_priors_dict(PTA,P)
   

    logging.info("Testing KF using parameters sampled from prior")
    params = priors.sample(1)


    print("The sampled params are as follows")
    print(params)


    model_likelihood = KF.likelihood(params)
    logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    
    # #Now run the Bilby sampler
    npoints=2000
    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/",npoints=npoints)
    logging.info("The run has completed OK")
