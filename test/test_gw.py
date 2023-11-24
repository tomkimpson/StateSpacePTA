#This is the test file for py_src/gravitational_waves.py 
from py_src import gravitational_waves #,system_parameters, pulsars 
import random 
import numpy as np 
from numpy import sin,cos

"""Check the principal axes function"""
def test_principal_axes():

    N = 5
    thetas = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    phis = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    psis = np.random.uniform(low=0.0,high=2*np.pi,size=N)

    for i in range(N):
        m,n = gravitational_waves.principal_axes(thetas[i],phis[i],psis[i])

        #Check magnitudes
        np.testing.assert_almost_equal(np.linalg.norm(m), 1.0) #https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html
        np.testing.assert_almost_equal(np.linalg.norm(n), 1.0) 

        #Check directions
        direction_inferred = np.cross(m,n)
        direction_explicit = np.array([sin(thetas[i])*cos(phis[i]),sin(thetas[i])*sin(phis[i]),cos(thetas[i])])
        np.testing.assert_almost_equal(direction_explicit,-direction_inferred)

"""Check the null model is all zeros as expected"""
def test_null_model():
    

    Npsr = 50
    Ntimes = 500
    H_factor = gravitational_waves.null_model(
                                np.random.uniform(), #delta
                                np.random.uniform(), #alpha
                                np.random.uniform(), #psi
                                np.random.uniform(size=Npsr), #this is q
                                np.random.uniform(), # q products
                                np.random.uniform(), #h
                                np.random.uniform(), #iota
                                np.random.uniform(), #omega
                                np.random.uniform(size=Ntimes), #t
                                np.random.uniform(), #phi0
                                np.random.uniform(), #chi
                            )
    assert np.all(H_factor==0)
