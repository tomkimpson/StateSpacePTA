

class Priors:



 def __init__(self,pulsar_parameters,GW_parameters):

        #GW parameters
        self.omega_gw=1e-7      
        self.phi0_gw= 0.20     
        self.psi_gw=  2.5
        self.iota_gw= 0.0
        self.delta_gw= 0.0
        self.alpha_gw= 1.0
        self.h=1e-8

        #pulsar_parameters
        self.f =pulsar_parameters.f
        self.fdot =pulsar_parameters.fdot
        self.d = pulsar_parameters.d
        self.gamma = pulsar_parameters.gamma

   
        #Noise parameters
        self.sigma_p= 1e-6
        self.sigma_m= 1e-13


