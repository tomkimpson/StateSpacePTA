from numpy import sin,cos 
import numpy as np 
from numba import jit,njit,guvectorize




"""
Calculate the principal axes vectors for each GW source. 
"""
@njit(fastmath=True)
def principal_axes(theta,phi,psi):

    
    m = np.zeros((len(theta),3)) #size K GW sources x 3 component directions

    m[:,0] = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)    # x-component
    m[:,1] = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta)) # y-component
    m[:,2] = sin(psi)*sin(theta)                                 # z-component


    n = np.zeros((len(theta),3)) #size K GW sources x 3 component directions
    n[:,0] = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n[:,1] = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n[:,2] = cos(psi)*sin(theta)
   

    return m,n

"""
Get the hplus and hcross amplitudes 
"""
@njit(fastmath=True)
def h_amplitudes(h,ι): 
    return h*(1.0 + cos(ι)**2),h*(-2.0*cos(ι)) #hplus,hcross



"""
Return the two polarisation tensors e_+, e_x
Reshapes allow vectorisation and JIT compatability 
"""
@njit(fastmath=True)
def polarisation_tensors(m, n):
    x, y = m.shape

    e_plus = m.reshape(x, 1, y) * m.reshape(1, x, y) - n.reshape(1, x, y) * n.reshape(x, 1, y)
    e_cross = m.reshape(x, 1, y) * n.reshape(1, x, y) - n.reshape(1, x, y) * m.reshape(x, 1, y)

    #i.e. return m[:, None] * m[None, :] - n[:, None] * n[None, :]. See https://stackoverflow.com/questions/77319805/vectorization-of-complicated-matrix-calculation-in-python

    return e_plus,e_cross


#There may be a faster way to do this. See again https://stackoverflow.com/questions/77319805/vectorization-of-complicated-matrix-calculation-in-python
@jit(nopython=True)
def h_summation(H, q):
    N, _ = q.shape
    _, _, K = H.shape
    result = np.zeros((N, K), dtype=H.dtype)
    
    for i in range(3):
        for j in range(3):
            result += H[i, j] * q[:, i:i+1] * q[:, j:j+1]
    
    return result


"""
This function is used to add two 2D matrices of different shapes
a(K,T)
b(K,N) 

It returns an array of shape (K,T,N)
"""
@njit
def add_matrices(a, b):
    K, T, N = a.shape[0], a.shape[1], b.shape[1]
    return a.reshape(K,T,1) + b.reshape(K,1,N)


"""
This function is used to add together a 3D array and a 2D array

The 2D array is added to each slice of the 3D array
"""
@njit
def add_slices(array3D, array2D):
    K,T = array2D.shape
    return array3D + array2D.reshape(K,T,1)

"""
This function is used to multiple together a 3D array and a 2D array

The 2D array multiplies each specified slice of the 3D array

array3D(K,T,N)
array2D(N,K)

"""
@njit
def multiply_slices(array3D, array2D):
 
    K,T,N = array3D.shape 

    tmp_array = array2D.T
    array3D * tmp_array.reshape(K,1,N)
    
    # out = np.empty_like(array3D)
    # for t in range(T):
    #     out[:,t,:] = array3D[:,t,:] * array2D.T
    # return out

    return array3D * tmp_array.reshape(K,1,N)



"""
What is the GW modulation factor, including all pulsar terms?
"""
@njit(fastmath=True)
def gw_psr_terms(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0,chi):

   
    
    m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi)      # Get the principal axes. Shape (K,3)
    gw_direction        = np.cross(m,n)                                    # And the direction of each source. This works for K sources. e.g. np.cross(m[i,:],n[i,:] ==gw_direction[i,:]
    e_plus,e_cross      = polarisation_tensors(m.T,n.T)                    # The polarization tensors. Shape (3,3,K)
    hp,hx               = h_amplitudes(h,iota)                             # plus and cross amplitudes. Shape (10,)
    Hij                 = hp * e_plus + hx * e_cross                       # amplitude tensor. Shape (3,3,K)
    hbar                = h_summation(Hij,q)                               # Calculate the sum H_ij q_i q^j. Shape (N,K)
    earth_term_phase    = np.outer(-omega,t) +  phi0.reshape(len(omega),1) # shape (N,t) # the phase term from the Earth-components
    dot_product         = 1.0 + np.dot(q,gw_direction.T)                   # matmul might be a bit faster, but np.dot has JIT support. Shape (N,K)
    pulsar_term_phase   = add_matrices(earth_term_phase,chi)               # Add (K,T) to (K,N) to produce (K,T,N)
    amplitude_term      = 0.50 * (hbar/dot_product)                        # Get the amplitude term. Shape(N,K)
    net_phase_term      = add_slices(cos(pulsar_term_phase), cos(earth_term_phase)) #Add together the pulsar term (K,T,N) and the earth term (K,T)
    
    GW_factor = np.sum(multiply_slices(net_phase_term, amplitude_term),axis=0) #shape (T,N)

    return GW_factor






def gw_earth_terms(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0,chi):
    #placeholder
    return 1.0





"""
The null model - i.e. no GW
"""
@njit(fastmath=True)
def null_model(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0,chi):
    return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    








#@njit(fastmath=True)
# def gw_prefactors(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):
#     #Get GW direction
#     m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi)    
#     gw_direction        = np.cross(m,n)

#     #Now get the strain amplitude 
#     #
#     # For e_+,e_x, Tensordot might be a bit faster, but list comprehension has JIT support
#     # Note these are 1D arrays, rather than the usual 2D struture
#     e_plus              = np.array([m[i]*m[j]-n[i]*n[j] for i in range(3) for j in range(3)]) 
#     e_cross             = np.array([m[i]*n[j]-n[i]*m[j] for i in range(3) for j in range(3)])
#     hp,hx               = h_amplitudes(h,iota) 
#     Hij                 = hp * e_plus + hx * e_cross

#     hbar                = np.dot(Hij,q_products) #length = Npsr
    

#     #Shared time dependent terms
#     earth_term_phase = -omega*t + phi0
    
#     #Define a dot product variable
#     dot_product         = 1.0 + np.dot(q,gw_direction) #matmul might be a bit faster, but np.dot has JIT support

#     return dot_product,hbar,earth_term_phase.reshape(len(t),1) #shapes [(Npsr,),(Npsr,),(Ntimes,1)]

