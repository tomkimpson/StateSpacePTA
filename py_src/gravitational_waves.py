from numpy import sin,cos 
import numpy as np 
from numba import jit 

"""
Given the GW parameters, the pulsar parameters and the time,
compute the frequency correction factor.
Returns an object of shape (n times, n pulsars)
"""







#### EXP FORM
@jit(nopython=True)
def gw_prefactor_optimised_exp(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):
        

        #GW direction axes
        m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi)   
        gw_direction        = np.cross(m,n)
        dot_product         = 1.0 + np.dot(q,gw_direction)                 # matmul might be a bit faster, but np.dot has JIT support

        #GW polarization tensors
        e_plus              = np.array([[m[i]*m[j]-n[i]*n[j] for i in range(3)] for j in range(3)]) #tensordot might be a bit faster, but list comprehension has JIT support
        e_cross             = np.array([[m[i]*n[j]-n[i]*m[j] for i in range(3)] for j in range(3)])

        #Plus and cross strain amplitudes
        hp,hx               = h_amplitudes(h,iota) 

        #The 3x3 Hij tensor, which is then flattened
        Hij                 = hp * e_plus + hx * e_cross
        Hij_flat            = Hij.flatten()

        #H_ij q^i q^j
        hbar                = np.dot(Hij_flat,q_products) #length = Npsr
        

        #h_ij q^i q^j evaluated at Earth
        time_variation = np.exp(1j*(-omega*t+phi0))
        hij = np.outer(time_variation,hbar) #This has shape(ntimes,npulsars)


        #The complete factor
        GW_factor = 1.0 - 0.50 * (hij/dot_product) * (1.0 - np.exp(1j*omega*dot_product*d))
        return np.real(GW_factor) #This has shape(n times, n pulsars)


#### OLD FORM
@jit(nopython=True)
def gw_prefactor_optimised_old(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):

        print("alpha = ", alpha)

        m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi)    
        gw_direction        = np.cross(m,n)
      
        dot_product         = 1.0 + np.dot(q,gw_direction) #matmul might be a bit faster, but np.dot has JIT support




        e_plus              = np.array([[m[i]*m[j]-n[i]*n[j] for i in range(3)] for j in range(3)]) #tensordot might be a bit faster, but list comprehension has JIT support
        e_cross             = np.array([[m[i]*n[j]-n[i]*m[j] for i in range(3)] for j in range(3)])

        hp,hx               = h_amplitudes(h,iota) 
        Hij                 = hp * e_plus + hx * e_cross
        Hij_flat            = Hij.flatten()

        hbar                = np.dot(Hij_flat,q_products)





        prefactor    = 0.5*(hbar / dot_product)*(1.0 - cos(omega*d*dot_product))


        tensor = np.outer(t,dot_product) #This has shape(n times, n pulsars)
        time_variation = cos(-omega*tensor + phi0)

        GW_factor = 1.0 - prefactor * time_variation
        return GW_factor #This has shape(n times, n pulsars)



#### TRIG FORM
@jit(nopython=True)
def gw_prefactor_optimised_trig(delta,alpha,psi,q,q_products,h,iota,omega,d,t,phi0):

     
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
        
        
        little_a = -omega*t + phi0
        little_b = omega*dot_product*d
        

        little_a = little_a.reshape((522,1))
        little_b = little_b.reshape((1,len(dot_product)))
        blob = little_a+little_b

        trig_block = cos(little_a).reshape((522,1)) - cos(blob)

        GW_factor = 1 - 0.50*(hbar/dot_product)*trig_block

        #h_ij q^i q^j evaluated at Earth
        #time_variation = np.exp(1j*(-omega*t+phi0))
        #hij = np.outer(time_variation,hbar) #This has shape(ntimes,npulsars)


        #The complete factor
        #GW_factor = 1.0 - 0.50 * (hij/dot_product) * (1.0 - np.exp(1j*omega*dot_product*d))
        #return np.real(GW_factor) #This has shape(n times, n pulsars)

        return GW_factor





gw_prefactor_optimised = gw_prefactor_optimised_trig



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




