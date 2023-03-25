print ("hello")



import dynesty
import numpy as np 
from numpy import linalg

ndim = 200  # number of dimensions
C = np.identity(ndim)  # set covariance to identity matrix
Cinv = linalg.inv(C)  # precision matrix
lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(linalg.det(C)))  # ln(normalization)




print(Cinv)

# 250-D iid standard normal log-likelihood
def loglikelihood(x):
    """Multivariate normal log-likelihood."""
    return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm