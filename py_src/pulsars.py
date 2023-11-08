
from numpy import sin, cos
import numpy as np 
import pandas as pd 
import logging
#from utils import get_project_root
from gravitational_waves import principal_axes

from pathlib import Path
import os 


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _unit_vector(theta,phi):

    qx = sin(theta) * cos(phi)
    qy = sin(theta) * sin(phi)
    qz = cos(theta)

    return np.array([qx, qy, qz]).T


# def _convert_vector_to_ra_dec(v):

#     x,y,z = v[0],v[1],v[2]

#     r = np.sqrt(x**2 + y**2 + z**2)

#     theta = np.arccos(z/r)
#     phi = np.arctan2(y,x)

#     return np.pi/2.0 - theta, phi #dec/ra



def create_pulsars(P):

    #F64, F32, longdouble etc
    #Other number formats are not well tested currently. Use F64 as standard for now
    NF = P["NF"]


    #Universal constants
    pc = NF(3e16)     # parsec in m
    c  = NF(3e8)      # speed of light in m/s


    #Load the pulsar data
    root = _get_project_root()
    pulsars = pd.read_csv(root / "data/NANOGrav_pulsars.csv")

    #Select a subset of pulsars
    if P["Npsr"] != 0:
        pulsars = pulsars.sample(P["Npsr"],random_state=P["seed"]) #can also use  pulsars.head(N) to sample  

        
    #Extract the parameters
    f         = pulsars["F0"].to_numpy(dtype=NF)
    fdot      = pulsars["F1"] .to_numpy(dtype=NF)
    d         = pulsars["DIST"].to_numpy(dtype=NF)*1e3*pc/c #this is in units of s^-1
    γ         = np.ones_like(f,dtype=NF) * 1e-13  #for every pulsar let γ be 1e-13
    δ         = pulsars["DECJD"].to_numpy()
    α         = pulsars["RAJD"].to_numpy()
    q         = _unit_vector(np.pi/2.0 -P["δ"], P["α"]) #3 rows, N columns
        
        
    #Create a flattened q-vector for optimised calculations later
    # i.e. given q = [qx,qy,qz], get the 9 (non-indep!) products qx*qx, qx*qy,...
    q_products = np.zeros((len(f),9))
    k = 0
    for n in range(len(f)):
        k = 0
        for i in range(3):
            for j in range(3):
                q_products[n,k] = q[n,i]*q[n,j]
                k+=1
    q_products = q_products.T


       
    #Get the principal axes for every GW source. 
    m,n                 = principal_axes(np.pi/2.0 - P["δ"],P["α"],P["ψ"]) # shape (K,3)
        
      
    #And now get the chi term, shape (K,N)
    gw_directions = np.cross(m,n)



    num_gw_sources = P["num_gw_sources"]
    Ω = P["Ω"]


    dot_product =  np.zeros((num_gw_sources,len(q)))
    chi =  np.zeros((num_gw_sources,len(q)))
    for i in range(num_gw_sources):
            dot_product[i,:]          = 1.0 + np.dot(q,gw_directions[i,:])
            chi[i,:]             = np.mod(Ω[i]*d*dot_product[i,:],2*np.pi)



    #Create some useful quantities

    dt      = P["cadence"] * 24*3600 #from days to step_seconds
    end_seconds  = P["T"]* 365*24*3600 #from years to second
    t       = np.arange(0,end_seconds,self.dt)
        
    #if σp is defined then set all pulsars with that value
    #else assign randomly within a range 
    generator = np.random.default_rng(SystemParameters["sigma_p_seed"])
    if SystemParameters["σp"] is None:
        σp = generator.uniform(low = 1e-21,high=1e-19,size=self.Npsr)
        logging.info("You are assigning the σp terms randomly")
    else:
        σp = np.full(P["Npsr"],P["σp"])


        #Rescaling
    ephemeris = f + np.outer(t,fdot) 
    fprime    = f - ephemeris[0,:] #this is the scaled state variable at t=0 
        


    #return a dictionary object
    output_dict = {}

    return output_dict
    #      










