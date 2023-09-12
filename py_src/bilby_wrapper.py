import bilby
import logging
logger = logging.getLogger(__name__).setLevel(logging.INFO)


"""Here is some test documentation"""
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):
        return self.model.likelihood(self.parameters)
    
#See https://lscsoft.docs.ligo.org/bilby/bilby-mcmc-guide.html  
def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

    #Run the sampler
    logging.info("Starting the bilby sampler")
    result = bilby.run_sampler(likelihood, priors, 
                               sampler ='bilby_mcmc',
                               label = label,
                               outdir=outdir,
                               check_point_plot=False,
                               nsamples=1000,  # This is the number of raw samples
                               thin_by_nact=0.2,  # This sets the thinning factor
                               ntemps=8,  # The number of parallel-tempered chains
                               npool=1,  # The multiprocessing cores to use
                               L1steps=100,  # The number of internal steps to take for each iteration
                               proposal_cycle='default')  # Use the standard (non-GW) proposal cycle
        

    return result
