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
            

def BilbySampler(KalmanFilter,init_parameters,priors):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    print("RUN THE SAMPLER")
    result = bilby.run_sampler(likelihood, priors, label = "test48",outdir=".",
                            sampler ='dynesty',check_point_plot=False,
                            sample='rwalk', walks=10, npoints=100,dlogz=5,
                            npool=6,plot=True,resume=False)

    return result