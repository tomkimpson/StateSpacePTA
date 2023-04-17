import numpy as np
from noisyopt import minimizeCompass

def obj(x):
    return (x**2).sum() + 0.1*np.random.randn()

res = minimizeCompass(obj, x0=[1.0, 2.0], deltatol=0.1, paired=False)


print (res)