

import numpy as np 
import logging

logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)







def specify_system_parameters(NF=np.float64,    # the number format of the arguments                 
                             T = 10,           # how long to integrate for in years
                             cadence=7,        # the interval between observations
                             Ω=   5.0e-7,        # GW angular frequency
                             Φ0 = 0.20,        # GW phase offset at t=0
                             ψ =  2.50,        # GW polarisation angle
                             ι = 1.0,          # GW source inclination
                             δ =  1.0,         # GW source declination
                             α =  1.0,         # GW source right ascension
                             h =  5e-13,        # GW plus strain
                             σp = 1e-13,       # process noise standard deviation
                             σm = 1e-11,        # measurement noise standard deviation
                             Npsr = 0,         # Number of pulsars to use in PTA. 0 = all
                             use_psr_terms_in_data=True, # when generating the synthetic data, include pulsar terms?
                             measurement_model='pulsar',# what do you want the KF measurement model to be? One of pulsar, earth,null
                             seed = 1237,       # this is the noise seed. It is used for sdeint and gaussian measurement noise
                             σp_seed=1234,      # this is the seed when we randomly generate simga_p parameter values
                             orthogonal_pulsars=False, #if True overwrite true RA/DEC of pulsars and create a ring of pulsars perpendicular to GW direction
                             num_gw_sources = 1 #how many GW sources are there on the sky?)
                            ):

    returned_dict = {
                        "NF": NF, 
                        "cadence": cadence,
                        "T":  NF(cadence), 
                        "Ω":  np.array([NF(Ω)]), 
                        "Φ0": np.array([NF(Φ0)]),
                        "ψ":  np.array([NF(ψ)]),
                        "ι":  np.array([NF(ι)]),
                        "δ":  np.array([NF(δ)]),
                        "α":  np.array([NF(α)]),
                        "h":  np.array([NF(h)]),
                        "σp": σp,#can be = None for random assignment. Handle NF conversion in pulsars.py
                        "σm":σm,
                        "Npsr":int(Npsr),
                        "use_psr_terms_in_data": use_psr_terms_in_data,
                        "measurement_model": measurement_model,
                        "seed": seed,
                        "sigma_p_seed":σp_seed,
                        "orthogonal_pulsars": orthogonal_pulsars,
                        "num_gw_sources":num_gw_sources
        }

    return returned_dict