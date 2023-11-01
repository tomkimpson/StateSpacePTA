#This is the test file for py_src/gravitational_waves.py 
from py_src import gravitational_waves #,system_parameters, pulsars 
import random 
import numpy as np 
from numpy import sin,cos
import jax.numpy as jnp 

"""Check the principal axes function"""
def test_principal_axes():
    K = 10
    thetas = np.random.uniform(low=0.0,high=np.pi,size=K)
    phis = np.random.uniform(low=0.0,high=2*np.pi,size=K)
    psis = np.random.uniform(low=0.0,high=2*np.pi,size=K)

    m,n = gravitational_waves.principal_axes(thetas,phis,psis)


    #Check the magnitudes
    np.testing.assert_array_almost_equal(np.linalg.norm(m,axis=1),np.ones(K)) #check
    np.testing.assert_array_almost_equal(np.linalg.norm(n,axis=1),np.ones(K)) #check


    #Check the direction
    direction_inferred = np.cross(m,n).T
    direction_explicit = np.array([sin(thetas)*cos(phis),sin(thetas)*sin(phis),cos(thetas)])
    np.testing.assert_almost_equal(direction_explicit,-direction_inferred,decimal=6)
    
   
"""Check the h amplitudes function"""
def test_h_amplitudes():
 
    #When iota is zero
    h = 1.0
    iota = 0.0

    hp,hx = gravitational_waves._h_amplitudes(h,iota)
    np.testing.assert_almost_equal(hp,2) 
    np.testing.assert_almost_equal(hx,-2) 


    #when iota is pi/2
    h = 1.0
    iota = np.pi/2

    hp,hx = gravitational_waves._h_amplitudes(h,iota)
    np.testing.assert_almost_equal(hp,1) 
    np.testing.assert_almost_equal(hx,0) 



    #when h is zero
    h = 0.0
    iota = 1.2345678 #this can be anything, since the strain is zero

    hp,hx = gravitational_waves._h_amplitudes(h,iota)
    np.testing.assert_almost_equal(hp,0) 
    np.testing.assert_almost_equal(hx,0) 



    #check vectors also work
    h = np.array([1.0,1.0])
    iota = np.array([0.0,np.pi/2])
    hp,hx = gravitational_waves._h_amplitudes(h,iota)
    np.testing.assert_almost_equal(hp,np.array([2,1]))
    np.testing.assert_almost_equal(hx,np.array([-2,0]))



def test_basis():


    #Test an analytical example with known solutions
    m = np.array([1,2,3])
    n = np.array([1,0,1])

    ep,ex = gravitational_waves._polarisation_tensors(m,n)

    analytical_ep = np.array([[0,2,2],[2,4,6],[2,6,8]])
    analytical_ex = np.array([[0,-2,-2],[2,0,2],[2,-2,0]])

    np.testing.assert_almost_equal(ep,analytical_ep)
    np.testing.assert_almost_equal(ex,analytical_ex)


