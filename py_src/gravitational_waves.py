from numpy import sin,cos 
import numpy as np 
from numba import jit 

@jit(nopython=True)
def gw_prefactors(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):

    m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi)    
    gw_direction        = np.cross(m,n)

    dot_product         = 1.0 + np.dot(q,gw_direction) #matmul might be a bit faster, but np.dot has JIT support


    e_plus              = np.array([[m[i]*m[j]-n[i]*n[j] for i in range(3)] for j in range(3)]) #tensordot might be a bit faster, but list comprehension has JIT support
    e_cross             = np.array([[m[i]*n[j]-n[i]*m[j] for i in range(3)] for j in range(3)])

    hp,hx               = h_amplitudes(h,iota) 
    Hij                 = hp * e_plus + hx * e_cross
    Hij_flat            = Hij.flatten()


    #H_ij q^i q^j
    hbar                = np.dot(Hij_flat,q_products) #length = Npsr


    #Shared time dependent terms
    little_a = -omega*t + phi0
    little_a = little_a.reshape((len(t),1)) #todo - avoid this reshape

    return dot_product,hbar,little_a



"""
What is the GW modulation factor, just for the earth terms
"""
@jit(nopython=True)
def gw_earth_terms(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):
    dot_product,hbar,little_a = gw_prefactors(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0)
    trig_block = cos(little_a).reshape((len(t),1)) 
    GW_factor = 0.50*(hbar/dot_product)*trig_block
    return GW_factor



"""
What is the GW modulation factor, including all pulsar terms?
"""
@jit(nopython=True)
def gw_psr_terms(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):

    dot_product,hbar,little_a = gw_prefactors(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0)



    #Extra pulsar terms
    little_b = omega*dot_product*d
    little_b = little_b.reshape((1,len(dot_product)))
    blob = little_a+little_b
    trig_block = cos(little_a).reshape((len(t),1)) - cos(blob)
    GW_factor = 0.50*(hbar/dot_product)*trig_block


    return GW_factor



"""
The null model - i.e. no GW
"""
@jit(nopython=True)
def null_model(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):
    return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    



@jit(nopython=True)
def principal_axes(theta,phi,psi):
    
    m1 = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)
    m2 = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta))
    m3 = sin(psi)*sin(theta)
    m = [m1,m2,m3]

    n1 = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n2 = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n3 = cos(psi)*sin(theta)
    n = [n1,n2,n3]

    return m,n

@jit(nopython=True)
def h_amplitudes(h,ι): 

    hplus = h*(1.0 + cos(ι)**2)
    hcross = h*(-2.0*cos(ι))

    return hplus,hcross










