

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
                 h =  5e-15,        # GW plus strain
                 σp = 1e-13,       # process noise standard deviation
                 σm = 1e-11,        # measurement noise standard deviation
                 Npsr = 0,         # Number of pulsars to use in PTA. 0 = all
                 use_psr_terms_in_data=True, # when generating the synthetic data, include pulsar terms?
                 measurement_model='pulsar',# what do you want the KF measurement model to be? One of pulsar, earth,null
                 seed = 1237,       # this is the noise seed. It is used for sdeint and gaussian measurement noise
                 σp_seed=1234,      # this is the seed when we randomly generate simga_p parameter values
                 orthogonal_pulsars=False, #if True overwrite true RA/DEC of pulsars and create a ring of pulsars perpendicular to GW direction
                 num_gw_sources = 1 #how many GW sources are there on the sky?
                 ): 

        logging.info("Welcome to the Kalman Filter Nested Sampler for PTA GW systems")

        self.NF = NF 
        self.T = NF(T) 
        self.cadence = NF(cadence)
        self.Ω = np.array([NF(Ω)])
        self.Φ0 = np.array([NF(Φ0)])
        self.ψ = np.array([NF(ψ)])
        self.ι = np.array([NF(ι)])
        self.δ = np.array([NF(δ)])
        self.α = np.array([NF(α)])
        self.h = np.array([NF(h)])
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



        if num_gw_sources == 2:
            logging.info("Running for two deterministic GW sources")
            
            self.Ω = np.array([5e-7,2e-8])
            self.Φ0 = np.array([0.20,1.50])
            self.ψ = np.array([2.50,0.35])
            self.ι = np.array([1.0,1.2])
            self.δ = np.array([1.0,0.70])
            self.α = np.array([1.0,1.30])
            self.h = np.array([5e-15,2e-15])


        if num_gw_sources == 5:
            logging.info("Running for five deterministic GW sources")
            
            self.Ω = np.array([5e-7,1e-7,2e-7,3e-7,4e-7])
            self.Φ0 = np.array([0.20,1.50,0.30,0.40,1.60])
            self.ψ = np.array([2.50,0.35,2.60,0.70,0.10])
            #self.ι = np.array([1.0,1.2,1.3,0.9,0.95])
            self.ι = np.array([1.0,1.0,1.0,1.0,1.0]) #all the same
            self.δ = np.array([1.0,0.70,1.1,0.8,0.9])
            self.α = np.array([1.0,1.30,0.8,0.9,1.2])
            #self.h = np.array([5e-15,2e-15,6e-15,7e-15,])
            self.h = np.array([5e-15,5e-15,5e-15,5e-15,5e-15]) #all the same, consistent SNR for every source?

        # if num_gw_sources > 1:
        #     logging.info("Multiple GW sources requested. Overwriting default GW parameters and randomly sampling")
        #     generator = np.random.default_rng(self.seed)

        #     self.Ω = generator.uniform(low = 1e-7,high=1e-6,size=self.num_gw_sources)
        #     self.Φ0 = generator.uniform(low = 0.0,high=np.pi/2,size=self.num_gw_sources)
        #     self.ψ = generator.uniform(low = 0.0,high=np.pi,size=self.num_gw_sources)
        #     self.ι = generator.uniform(low = 0.0,high=np.pi/2,size=self.num_gw_sources)
        #     self.δ = generator.uniform(low = 0.0,high=np.pi/2,size=self.num_gw_sources)
        #     self.α = generator.uniform(low = 0.0,high=np.pi,size=self.num_gw_sources)
        #     self.h = generator.uniform(low = 5e-15,high=1e-14,size=self.num_gw_sources)

        
        #     logging.info("Selected random GW parameters are as follows:")

        #     logging.info(f"Omega = {self.Ω}")
        #     logging.info(f"Phi0 = {self.Φ0}")
        #     logging.info(f"psi = {self.ψ}")
        #     logging.info(f"iota = {self.ι}")
        #     logging.info(f"delta = {self.δ}")
        #     logging.info(f"alpha = {self.α}")
        #     logging.info(f"h = {self.h}")
        #     logging.info("***END SYSTEM PARAMETERS***")

