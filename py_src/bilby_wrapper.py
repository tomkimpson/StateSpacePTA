import bilby
import sys
import numpy as np 
import sys
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
    print("RUN THE SAMPLER")

   # https://lscsoft.docs.ligo.org/bilby/api/bilby.bilby_mcmc.sampler.Bilby_MCMC.html
    # result = bilby.run_sampler(likelihood,priors,
	#   	                       label=label,
    #                             outdir=outdir,
	#   		                    sampler="bilby_mcmc",
	#   		                    npool=32,
	#   		                    ntemps=32,
    #                             #nensemble = 10,
    #                             nsamples=500,
    #                             resume=False,
    #                             diagnostic=True,
    #                             #stop_after_convergence=True,
    #                             #initial_sample_method="maximise",
    #                            # thin_by_nact=1
	#  		       )

    result = bilby.run_sampler(likelihood, priors, 
                              label = label,
                              outdir=outdir,
                              sampler ='dynesty', #sampler=bilby_mcmc, dynesty,ultranest,pymultinest
			                  sample='rwalk_dynesty',
                              check_point_plot=False,
                              npoints=2000,
                              dlogz=1e-3,
			                  #logz=1e-4,
                              npool=32,
			                  plot=False,resume=False)

    return result
