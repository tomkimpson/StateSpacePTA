

from numpy import sin,cos 
import numpy as np 
class GWs:





    def __init__(self,P):


        m,n                 = principal_axes(np.pi/2.0 - P["delta_gw"],P["alpha_gw"],P["psi_gw"])    
        self.n              = np.cross(m,n)            
        hp,hx               = h_amplitudes(P["h"],P["iota_gw"]) 
        e_plus              = np.tensordot(m, m, axes=0) - np.tensordot(n, n, axes=0)
        e_cross             = np.tensordot(m, n, axes=0) - np.tensordot(n, m, axes=0) 
        self.Hij            = hp * e_plus + hx * e_cross




        #assing some quantities to self
        self.omega_gw = P["omega_gw"]
        self.phi0_gw = P["phi0_gw"]
        self.psi_gw = P["psi_gw"]
        self.iota_gw = P["iota_gw"]
        self.delta_gw = P["delta_gw"]
        self.alpha_gw = P["alpha_gw"]
        self.h = P["h"]

def gw_prefactor(n,q, Hij,ω, d):

    dot_product  = 1.0 + np.matmul(q,n)
    hbar         =    [np.einsum('ij,i,j->', Hij, q[k,:], q[k,:]) for k in range(len(q))]
    ratio        = hbar / dot_product
    Hcoefficient = 1.0 - cos(ω*d*dot_product)
    prefactor    = 0.5*ratio*Hcoefficient

    return prefactor,dot_product

def gw_modulation(t,omega,phi0,prefactor,dot_product):
    time_variation = cos(-omega*t *dot_product + phi0)
    GW_factor = 1.0 - prefactor * time_variation
    return GW_factor

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

def h_amplitudes(h,ι): 


    hplus = h*(1.0 + cos(ι)**2)
    hcross = h*(-2.0*cos(ι))

    return hplus,hcross

