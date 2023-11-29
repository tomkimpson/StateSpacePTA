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
    
            
def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

    #Run the sampler
    logging.info("Starting the bilby sampler")
    result = bilby.run_sampler(likelihood, priors, 
                              label = label,
                              outdir=outdir,
                              sampler ='dynesty',
                              bound = 'single',
                              sample='rwalk_dynesty',
                              check_point_plot=False,
                              npoints=2000,
                              dlogz=0.1,
                              npool=1,
                              plot=False,resume=True)

    return result