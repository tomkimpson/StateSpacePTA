import bilby
import logging
logger = logging.getLogger(__name__).setLevel(logging.INFO)
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):

 
        ll,xres,yres = self.model.likelihood(self.parameters)
        
        return ll
    
            
def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    logging.info("Starting the bilby sampler")
    result = bilby.run_sampler(likelihood, priors, 
                              label = label,
                              outdir=outdir,
                              sampler ='dynesty',
			                  sample='rwalk_dynesty',
                              check_point_plot=False,
                              npoints=2500,
                              dlogz=1e-6,
                              npool=1,
			                  plot=False,resume=False)

    return result
