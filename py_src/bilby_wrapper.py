import bilby
import logging
logger = logging.getLogger(__name__).setLevel(logging.INFO)

import sys
"""Here is some test documentation"""
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):
        return self.model.likelihood(self.parameters)
    
            
def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir,npoints):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

    #Run the sampler
    logging.info("Starting the bilby sampler")
    result = bilby.run_sampler(likelihood, priors, 
                              label = label,
                              sampler='bilby_mcmc',
                              outdir=outdir,
                              nsamples = 1000,
                              thin_by_nact=0.2,
                              ntemps=8,
                              npool=1,
                              L1steps=100,
                              proposal_cyle='default')
    return result