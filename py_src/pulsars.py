
from numpy import sin, cos
import numpy as np 
import pandas as pd 
class Pulsars:


    def __init__(self,SystemParameters):



        NF = SystemParameters["NF"]

        pc = NF(3e16)     # parsec in m
        c  = NF(3e8)      # speed of light in m/s

        pulsars = pd.read_csv("../data/NANOGrav_pulsars.csv")
        if SystemParameters["Npsr"] != 0:
            #pulsars = pulsars.head(SystemParameters["Npsr"]) #can also use  pulsars.sample(N) to randonly sample 
            pulsars = pulsars.sample(SystemParameters["Npsr"],random_state=1) #can also use  pulsars.sample(N) to randonly sample 

        
        #Extract the parameters
        self.f = pulsars["F0"].to_numpy(dtype=NF)
        self.fdot = pulsars["F1"] .to_numpy(dtype=NF)
        self.fdot = np.zeros_like(self.fdot)# pulsars["F1"] .to_numpy(dtype=NF)

        self.d = pulsars["DIST"].to_numpy(dtype=NF)*1e3*pc/c #this is in units of s^-1
        #self.gamma = np.ones_like(self.f,dtype=NF) * 1e-13  #for every pulsar let γ be 1e-13
        self.gamma = np.zeros_like(self.f,dtype=NF) #* 1e-13  #for every pulsar let γ be 1e-13

   

        self.δ = pulsars["DECJD"].to_numpy()
        self.α = pulsars["RAJD"].to_numpy()
        self.q = unit_vector(np.pi/2.0 -self.δ, self.α) #3 rows, N columns
        

        #Create a flattened q-vector for optimised calculations later
        self.q_products = np.zeros((len(self.f),9))
        k = 0
        for n in range(len(self.f)):
            k = 0
            for i in range(3):
                for j in range(3):
                    self.q_products[n,k] = self.q[n,i]*self.q[n,j]
                    k+=1
        self.q_products = self.q_products.T

        #Assign some other useful quantities to self
        self.dt      = SystemParameters["cadence"] * 24*3600 #from days to step_seconds
        end_seconds  = SystemParameters["T"]* 365*24*3600 #from years to second
        self.t       = np.arange(0,end_seconds,self.dt)
        self.sigma_p =  SystemParameters["sigma_p"] 
        self.sigma_m =  SystemParameters["sigma_m"]
        self.Npsr    = len(self.f)
        self.NF = NF 



def unit_vector(theta,phi):

    qx = sin(theta) * cos(phi)
    qy = sin(theta) * sin(phi)
    qz = cos(theta)

    return np.array([qx, qy, qz]).T



#def convert_dec_to_theta(dec):

    
