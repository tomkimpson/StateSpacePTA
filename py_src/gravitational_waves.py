#from numpy import sin,cos 
#import numpy as np 
# from numba import jit,njit,prange
import sys
import jax.numpy as np
from jax import jit



"""
Calculate the principal axes vectors for each GW source. 
"""
@jit
def principal_axes(theta,phi,psi):


    #numpy is row major, as are jax np arrays
    #therefore we want to do operations on the rows, not the columns where possible

    m = np.zeros((3,len(theta))) # 3 rows, K columns
    
    m = m.at[0,:].set(np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))  # x-component
    m = m.at[1,:].set(-np.cos(phi)*np.cos(psi) - np.sin(psi)*np.sin(phi)*np.cos(theta)) # y-component
    m = m.at[2,:].set(np.sin(psi)*np.sin(theta))                                        # z-component

    n = np.zeros_like(m) 

    n = n.at[0,:].set(-np.sin(phi)*np.sin(psi) - np.cos(psi)*np.cos(phi)*np.cos(theta))
    n = n.at[1,:].set(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.cos(theta))
    n = n.at[2,:].set(np.cos(psi)*np.sin(theta))

    return m.T,n.T


@jit
def _h_amplitudes(h,ι): 
    return h*(1.0 + np.cos(ι)**2),h*(-2.0*np.cos(ι)) 



@jit
def _polarisation_tensors(m,n):
    e_plus = m[:, None] * m[None, :] - n[:, None] * n[None, :]
    e_cross = m[:, None] * n[None, :] - n[:, None] * m[None, :]
    return e_plus,e_cross




"""
This function is used to add two 2D matrices of different shapes
a(K,T)
b(K,N) 

It returns an array of shape (K,T,N)
"""
@jit
def add_matrices(a, b):
    K, T, N = a.shape[0], a.shape[1], b.shape[1]
    return a.reshape(K,T,1) + b.reshape(K,1,N)




"""
What is the GW modulation factor, including all pulsar terms?
"""
@jit
def gw_psr_terms(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,chi):
    K,N,T                    = len(delta),len(q),len(t)  #dimensions
   
    #Time -independent terms
    m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi) # Get the principal axes. Shape (K,3)
    gw_direction        = np.cross(m,n)                               # The direction of each source. Shape (K,)
    e_plus,e_cross      = _polarisation_tensors(m.T,n.T)              # The polarization tensors. Shape (3,3,K)
    hp,hx               = _h_amplitudes(h,iota)                       # plus and cross amplitudes. Shape (K,)
    Hij                 = hp * e_plus + hx * e_cross                  # amplitude tensor. Shape (3,3,K)
    Hij                 = Hij.reshape(K,9)                            # reshape it to enable dot product with q_products
    hbar                = np.dot(Hij,q_products)                      # Shape (K,N)
    dot_product         = 1.0 + q @ gw_direction.reshape(3,K)         # Shape (N,K)
  
    
    #Time-dependent terms
    #reshapes for broadcasting
    earth_term_phase    = (np.outer(-omega,t) +  phi0.reshape(len(omega),1)).reshape(len(t),K)      # i.e. -\Omega *t + \Phi_0
    pulsar_term_phase   = earth_term_phase.reshape(K,T,1) + chi.reshape(K,1,N)                      # i.e. -\Omega *t + \Phi_0 + \Chi

    #Bring it all together
    net_time_dependent_term = np.cos(earth_term_phase).reshape(K,T,1) - np.cos(pulsar_term_phase)
    amplitude               = 0.50*hbar/dot_product.reshape(hbar.shape)



    GW_factor = np.sum(net_time_dependent_term*amplitude.reshape(K,1,N),axis=0) #shape (T,N) #sum over K sources. GWs linearly superpose
    return GW_factor






"""
The null model - i.e. no GW
"""
# @njit(fastmath=True)
def null_model(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,chi):
    return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    


"""
not yet defns
"""
def gw_earth_terms(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,chi):
    sys.exit('Earth terms not set up propery yet')
    return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    










