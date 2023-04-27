

import numpy as np
from numba import jit,config
from numpy import sin,cos 

from system_parameters import disable_JIT
config.DISABLE_JIT = disable_JIT
import scipy.linalg as linalg

from gravitational_waves import gw_modulation



class EKF:

    """
    An EFF model of the state evolution
    
    """

    """
    The diagonal F matrix as a vector
    """
    # @jit(nopython=True)
    def F_function(gamma,dt):

        gw_block = np.eye(2)
        gw_block[0,1] = -dt
        
        f_block = np.diag(np.exp(-gamma*dt))
        

        return linalg.block_diag(gw_block,f_block) #This is a square matrix


    """
    The control vector
    """
    # @jit(nopython=True)
    def T_function(f0,fdot,gamma,t,dt):

       
        gw_block = np.zeros(2)
        
        
        fdot_time = t * fdot
        f_block = f0 + fdot_time + fdot*dt - np.exp(-gamma*dt)*(f0+fdot_time)


       
        return np.concatenate((gw_block,f_block),axis=0)
    



    def h_function(x,A,phase_i):

        Npsr = int((len(x) - 2.0))
        gw_phase = x[0]
        f = x[2:2+Npsr]
        
        GW_factor = gw_modulation(A,gw_phase,phase_i)

        return f*GW_factor
    


    def H_function(x,A,phase_i):

        Npsr = int((len(x) - 2.0))
        gw_phase = x[0]
        f = x[2:2+Npsr]
       



        dh_dphase = f*A*(sin(gw_phase)+sin(phase_i - gw_phase))
        dh_df = gw_modulation(A,gw_phase,phase_i)
       

        H = np.zeros((len(f),len(x))) #number of observations x number of states

        H[:,0] = dh_dphase

        for i in range(len(f)):
            H[i,i+2] = dh_df[i]


        return H




    """
    The  Q matrix 
    """
    # @jit(nopython=True)
    def Q_function(gamma,sigma_p,dt):


        gw_block = np.zeros((2,2))

        value = -sigma_p**2 * (np.exp(-2.0*gamma* dt) - 1.) / (2.0 * gamma)
        f_block = np.diag(value)

        

        return linalg.block_diag(gw_block,f_block) #This is a square matrix

 
     

    """
    The R matrix as a scalar
    """
    # @jit(nopython=True)
    def R_function(sigma_m,N):
        return np.eye(N)*sigma_m**2
     

