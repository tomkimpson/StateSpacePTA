

import numpy as np 
import logging

logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)

"""
Class of parameters which define the system
"""
class SystemParameters:


    def __init__(self,
                 NF=np.float64,    # the number format of the arguments
                 T = 10,           # how long to integrate for in years
                 cadence=7,        # the interval between observations
                 Ω=   5.0e-7,        # GW angular frequency
                 Φ0 = 0.20,        # GW phase offset at t=0
                 ψ =  2.50,        # GW polarisation angle
                 ι = 1.0,          # GW source inclination
                 δ =  1.0,         # GW source declination
                 α =  1.0,         # GW source right ascension
                 h =  1e-2,        # GW plus strain
                 σp = 1e-13,       # process noise standard deviation
                 σm = 1e-8,        # measurement noise standard deviation
                 Npsr = 0,         # Number of pulsars to use in PTA. 0 = all
                 use_psr_terms_in_data=True, # when generating the synthetic data, include pulsar terms?
                 measurement_model='pulsar',# what do you want the KF measurement model to be? One of pulsar, earth,null
                 seed = 1234,       # this is the noise seed. It is used for sdeint and gaussian measurement noise
                 σp_seed=1234,      # this is the seed when we randomly generate simga_p parameter values
                 orthogonal_pulsars=False, #if True overwrite true RA/DEC of pulsars and create a ring of pulsars perpendicular to GW direction
                 num_gw_sources = 1 #how many GW sources do we want?
                 ): 

        logging.info("Welcome to the Kalman Filter Nested Sampler for PTA GW systems")

        self.NF = NF 
        self.T = NF(T) 
        self.cadence = NF(cadence)
        self.Ω = NF(Ω)
        self.Φ0 = NF(Φ0)
        self.ψ = NF(ψ)
        self.ι = NF(ι)
        self.δ = NF(δ)
        self.α = NF(α)
        self.h = NF(h)
        self.σp = σp #can be = None for random assignment. Handle NF conversion in pulsars.py

        self.σm = NF(σm)
        self.Npsr = int(Npsr)

        self.use_psr_terms_in_data = use_psr_terms_in_data 
        self.measurement_model = measurement_model
        self.seed = seed
        self.sigma_p_seed = σp_seed
        self.orthogonal_pulsars =orthogonal_pulsars

        self.num_gw_sources = num_gw_sources

        logging.info(f"Random seed is {self.seed}")
        if σp ==1.0:
            logging.info("σp = 1.0 is a special value. \n Assigning process noise amplitudes randomly within a range. \n Please see synthetic_data.py")



        if self.num_gw_sources > 0: #always
            logging.info("Multiple GW sources requested. Overwriting default GW parameters and randomly sampling")
            generator = np.random.default_rng(self.seed)

            self.Ω = generator.uniform(low = 1e-7,high=1e-6,size=self.num_gw_sources)
            self.Φ0 = generator.uniform(low = 0.0,high=np.pi/2,size=self.num_gw_sources)
            self.ψ = generator.uniform(low = 0.0,high=np.pi,size=self.num_gw_sources)
            self.ι = generator.uniform(low = 0.0,high=np.pi/2,size=self.num_gw_sources)
            self.δ = generator.uniform(low = 0.0,high=np.pi/2,size=self.num_gw_sources)
            self.α = generator.uniform(low = 0.0,high=np.pi,size=self.num_gw_sources)
            self.h = generator.uniform(low = 1e-15,high=1e-14,size=self.num_gw_sources)


        print(f"Running with {len(self.Ω)} GW sources ")


        

        



        
    

