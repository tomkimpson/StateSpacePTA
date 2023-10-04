from numpy import sin,cos 
import numpy as np 
from numba import jit,njit 

#@njit(fastmath=True)
def gw_prefactors(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):
    #Get GW direction
    m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi)    
    gw_direction        = np.cross(m,n)

    #Now get the strain amplitude 
    #
    # For e_+,e_x, Tensordot might be a bit faster, but list comprehension has JIT support
    # Note these are 1D arrays, rather than the usual 2D struture
    e_plus              = np.array([m[i]*m[j]-n[i]*n[j] for i in range(3) for j in range(3)]) 
    e_cross             = np.array([m[i]*n[j]-n[i]*m[j] for i in range(3) for j in range(3)])
    hp,hx               = h_amplitudes(h,iota) 
    Hij                 = hp * e_plus + hx * e_cross
    hbar                = np.dot(Hij,q_products) #length = Npsr
    

    #Shared time dependent terms
    earth_term_phase = -omega*t + phi0
    
    #Define a dot product variable
    dot_product         = 1.0 + np.dot(q,gw_direction) #matmul might be a bit faster, but np.dot has JIT support

    return dot_product,hbar,earth_term_phase.reshape(len(t),1) #shapes [(Npsr,),(Npsr,),(Ntimes,1)]



"""
What is the GW modulation factor, just for the earth terms
"""
#@njit(fastmath=True)
def gw_earth_terms(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0,chi):
    dot_product,hbar,earth_term_phase = gw_prefactors(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0)
    GW_factor = 0.50*(hbar/dot_product)*(cos(earth_term_phase))
    return GW_factor


"""
What is the GW modulation factor, including all pulsar terms?
"""
#@njit(fastmath=True)
def gw_psr_terms(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0,chi):

    #print("DEBUG: here is the psr terms gw func")
    num_gw_sources = len(delta)
    GW_factor = np.zeros((len(t),len(d))) #times x NPSR 

   # print("num gw sources = ", num_gw_sources)
    for k in range(num_gw_sources):
        # print("for k = ", k)
        # print("delta",delta[k]),
        # print("alpha",alpha[k]),
        # print("psi",psi[k]),
        # print("H",h[k]),
        # print("iota",iota[k]),
        # print("omega",omega[k]),
        # print("phi0",phi0[k])
        dot_product,hbar,earth_term_phase = gw_prefactors(delta[k],alpha[k],psi[k],q,q_products,h[k],iota[k],omega[k],d,t,phi0[k])
        
        # print("got the dot product")
        # print("chi:", chi)
        # print(chi[k,:])
        # print(earth_term_phase)
        GW_factor += 0.50*(hbar/dot_product)*(cos(earth_term_phase) - cos(earth_term_phase +chi[k,:]))
       
   
    return GW_factor



"""
The null model - i.e. no GW
"""
# @njit(fastmath=True)
def null_model(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):
    return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    

# @njit(fastmath=True)
def principal_axes(theta,phi,psi):
    
    m1 = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)
    m2 = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta))
    m3 = sin(psi)*sin(theta)
    m = [m1,m2,m3]

    n1 = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n2 = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n3 = cos(psi)*sin(theta)
    n = [n1,n2,n3]

    return np.array(m),np.array(n)

# @njit(fastmath=True)
def h_amplitudes(h,ι): 
    return h*(1.0 + cos(ι)**2),h*(-2.0*cos(ι)) #hplus,hcross










