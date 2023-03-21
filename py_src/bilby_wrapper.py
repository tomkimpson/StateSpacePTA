import bilby
import sys
import numpy as np 
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):

        try:
            ll, xres, P = self.model.likelihood(self.parameters)
        except np.linalg.LinAlgError:
            ll= -np.inf
        if np.isnan(ll):
            ll = -np.inf
        return ll
            

def BilbySampler(KalmanFilter,init_parameters,injection_parameters,priors,label,outdir):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    print("RUN THE SAMPLER")
    result = bilby.run_sampler(likelihood, priors, injection_parameters = injection_parameters,label = label,outdir=outdir,
                            sampler ='dynesty',check_point_plot=False,
                            sample='rwalk', walks=10, npoints=100,
                            npool=6,plot=True,resume=False)

    return result