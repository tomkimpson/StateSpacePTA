
from numpy import sin, cos
import numpy as np 
import pandas as pd 
import logging
#from utils import get_project_root
from gravitational_waves import principal_axes

from pathlib import Path
import os 


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Pulsars:


    def __init__(self,SystemParameters):


        #F64, F32, longdouble etc
        #Not well tested currently. Use F64 as standard fow now
        NF = SystemParameters.NF


        #Universal constants
        pc = NF(3e16)     # parsec in m
        c  = NF(3e8)      # speed of light in m/s


        #Load the pulsar data
        root = get_project_root()
        pulsars = pd.read_csv(root / "data/NANOGrav_pulsars.csv")

        #Select a subset of pulsars
        if SystemParameters.Npsr != 0:
            pulsars = pulsars.sample(SystemParameters.Npsr,random_state=SystemParameters.seed) #can also use  pulsars.head(N) to sample  

        
        #Extract the parameters
        self.f         = pulsars["F0"].to_numpy(dtype=NF)
        self.fdot      = pulsars["F1"] .to_numpy(dtype=NF)
        self.d         = pulsars["DIST"].to_numpy(dtype=NF)*1e3*pc/c #this is in units of s^-1
        self.γ         = np.ones_like(self.f,dtype=NF) * 1e-13  #for every pulsar let γ be 1e-13
        self.δ         = pulsars["DECJD"].to_numpy()
        self.α         = pulsars["RAJD"].to_numpy()
        self.q         = unit_vector(np.pi/2.0 -self.δ, self.α) #3 rows, N columns
        
        
        #Create a flattened q-vector for optimised calculations later
        # i.e. given q = [qx,qy,qz], get the 9 (non-indep!) products qx*qx, qx*qy,...
        self.q_products = np.zeros((len(self.f),9))
        k = 0
        for n in range(len(self.f)):
            k = 0
            for i in range(3):
                for j in range(3):
                    self.q_products[n,k] = self.q[n,i]*self.q[n,j]
                    k+=1
        self.q_products = self.q_products.T


       
        #Get the principal axes for every GW source. 
        m,n                 = principal_axes(np.pi/2.0 - SystemParameters.δ,SystemParameters.α,SystemParameters.ψ) # shape (K,3)
        
      
        #And now get the chi term, shape (K,N)
        gw_directions = np.zeros((SystemParameters.num_gw_sources,3))
        dot_product =  np.zeros((SystemParameters.num_gw_sources,len(self.q)))
        self.chi =  np.zeros((SystemParameters.num_gw_sources,len(self.q)))
        for i in range(SystemParameters.num_gw_sources):
            gw_directions[i,:]        = np.cross(m[i,:],n[i,:])
            dot_product[i,:]          = 1.0 + np.dot(self.q,gw_directions[i,:])
            self.chi[i,:]             = np.mod(SystemParameters.Ω[i]*self.d*dot_product[i,:],2*np.pi)



        #Assign some other useful quantities to self
        #Some of these are already defined in SystemParameters, but I don't want to pass
        #the SystemParameters class to the Kalman filter - it should be completely blind
        #to the true parameters - it only knows what we tell it!
        self.dt      = SystemParameters.cadence * 24*3600 #from days to step_seconds
        end_seconds  = SystemParameters.T* 365*24*3600 #from years to second
        self.t       = np.arange(0,end_seconds,self.dt)
        self.Npsr    = len(self.f) 
        
        #if σp is defined then set all pulsars with that value
        #else assign randomly within a range 
        generator = np.random.default_rng(SystemParameters.sigma_p_seed)
        if SystemParameters.σp is None:
            self.σp = generator.uniform(low = 1e-21,high=1e-19,size=self.Npsr)
            logging.info("You are assigning the σp terms randomly")
        else:
            self.σp = np.full(self.Npsr,SystemParameters.σp)

        self.σm =  SystemParameters.σm
        self.NF = NF 

        #Rescaling
        self.ephemeris = self.f + np.outer(self.t,self.fdot) 
        self.fprime    = self.f - self.ephemeris[0,:] #this is the scaled state variable at t=0 
        


def unit_vector(theta,phi):

    qx = sin(theta) * cos(phi)
    qy = sin(theta) * sin(phi)
    qz = cos(theta)

    return np.array([qx, qy, qz]).T


def convert_vector_to_ra_dec(v):

    x,y,z = v[0],v[1],v[2]

    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)

    return np.pi/2.0 - theta, phi #dec/ra
