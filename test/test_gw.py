#This is the test file for py_src/gravitational_waves.py 
from py_src import gravitational_waves
import random 
import numpy as np 
from numpy import sin,cos

"""Check that the magnitude of the principal axes is always unity"""
def test_magnitude_of_principal_axes():

    N = 5
    thetas = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    phis = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    psis = np.random.uniform(low=0.0,high=2*np.pi,size=N)

    for i in range(N):
        m,n = gravitational_waves.principal_axes(thetas[i],phis[i],psis[i])

        np.testing.assert_almost_equal(np.linalg.norm(m), 1.0) #https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html
        np.testing.assert_almost_equal(np.linalg.norm(n), 1.0) 


        direction_inferred = np.cross(m,n)
        direction_explicit = np.array([sin(thetas[i])*cos(phis[i]),sin(thetas[i])*sin(phis[i]),cos(thetas[i])])

        #
        np.testing.assert_almost_equal(direction_explicit,direction_inferred)


# """ k \cross l should be the same as the direction unit vector"""
# def test_principal_axes_cross_product():

#     N = 1