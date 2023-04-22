import bilby
import sys
import numpy as np 
import sys
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):

        try:
            #ll, xres, P = self.model.likelihood(self.parameters)
            ll = self.model.likelihood(self.parameters)

        except np.linalg.LinAlgError:
            ll= -np.inf
        if np.isnan(ll):
            ll = -np.inf

        return ll
            

def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    print("RUN THE SAMPLER")


    # result = bilby.run_sampler(likelihood,priors,
	# 	               label=label,
    #                            outdir=outdir,
	# 		       sampler="bilby_mcmc",
	# 		       npool=32,
	# 		       ntemps=32,
    #                            nsamples=2000,
    #                            thin_by_nact=1
	# 		       )

    result = bilby.run_sampler(likelihood, priors, 
                               label = label,
                               outdir=outdir,
                               sampler ='dynesty', #sampler=bilby_mcmc, dynesty
			                   sample='rwalk',
                               check_point_plot=False,
                               npoints=500,
			                   dlogz=1e-4,
                               npool=32,
			                   plot=True,resume=False)







    # result = bilby.run_sampler(likelihood, priors, label = label,outdir=outdir,
    #                         sampler ='dynesty',check_point_plot=False,
    #                         sample='act-walk', walks=50, npoints=100,dlogz=0.10,
    #                         npool=4,plot=False,resume=False)

    return result
