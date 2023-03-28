

import numpy as np 
from scipy.integrate import odeint,solve_ivp



#system parameters
T = 10           # how long to integrate for in years
cadence=7        # the interval between observations
Ω=   5e-7           # GW angular frequency
Φ0 = 0.20        # GW phase offset at t=0
ψ =  2.50         # GW polarisation angle
ι =  1.0          # GW source inclination
δ =  1.0          # GW source declination
α =  1.0          # GW source right ascension
h =  1e-2         # GW strain
σp = 1e-8       # process noise standard deviation
σm = 1e-10        # measurement noise standard deviation
Npsr = 0          # Number of pulsars to use in PTA. 0 = all



#timing setup
dt = cadence*24*3600 
trange = np.arange(0,T*365*24*3600,dt)




#get one pulsar 
f0 = 100 #Hz 
fdot = -1e-15
gamma = 1e-13
d = 1*1e3*3e16 # 1 kpc in m 
d = d / 3e8 #now the distance is in units of s

psr_dec = 2.0
psr_ra = 2.0 


from numpy import sin, cos
def unit_vector(theta,phi):

    qx = sin(theta) * cos(phi)
    qy = sin(theta) * sin(phi)
    qz = cos(theta)

    return np.array([qx, qy, qz]).T


psr_q = unit_vector(np.pi/2.0 -psr_dec, psr_ra)





def lorenz(t, y): 
    return [-gamma * y+ gamma*(f0 + fdot*t) + fdot] 


y0 = [f0]
t_span = (0.0, trange[-1])

sol = solve_ivp(lorenz, t_span, y0,t_eval = trange)




intrinsic_frequency = sol.y.flatten()



# import matplotlib.pyplot as plt 
# plt.plot(sol.t,sol.y.flatten())
# plt.show()
# print(len(sol.t))


m,n = 




















#print(sol)

    # P   = SystemParameters(Npsr=2)       #define the system parameters as a class
    # PTA = Pulsars(P)               #setup the PTA
    # #GW  = GWs(P)                   #setup GW related constants and functions. This is a dict, not a class, for interaction later with Bilby 
    # data = SyntheticData(PTA,P) #generate some synthetic data


    # #Define the model 
    # model = LinearModel

    # #Initialise the Kalman filter
    # KF = KalmanFilter(model,data.f_measured,PTA)

   
    # true_parameters = priors_dict(PTA,P)
    # model_likelihood = KF.likelihood(true_parameters)
    # print("The log likelihood with the true parameters is: ", model_likelihood)
