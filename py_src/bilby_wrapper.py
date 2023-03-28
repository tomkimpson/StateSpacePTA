import bilby
import sys
import numpy as np 
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):

        
        try:
            ll = self.model.likelihood(self.parameters)
            #print("log likelihood = ", ll )

            
        except np.linalg.LinAlgError:
            ll= -np.inf
        if np.isnan(ll):
            ll = -np.inf

        return ll
    


    # def noise_log_likelihood(self):
        
    #     try:
    #         ll = self.model.null_likelihood(self.parameters)
            
    #     except np.linalg.LinAlgError:
    #         ll= -np.inf
    #     if np.isnan(ll):
    #         ll = -np.inf

    #     return ll
            

def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir,dlogz):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    print("RUN THE SAMPLER")
    result = bilby.run_sampler(likelihood, priors, 
                             label = label,
                             outdir=outdir,
                            sampler ='dynesty',
                            check_point_plot=False,
                            sample='auto', 
                            #walks=10, 
                            npoints=200,dlogz=dlogz,
                            npool=6, plot=True,resume=False)

    return result