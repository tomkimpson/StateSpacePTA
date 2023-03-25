import numpy as np

# Define the Van der Pol oscillator model
mu = 1.0

def f(x, u):
    # Define the state update function
    x_next = np.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0] + u
    ])
    return x_next

def h(x):
    # Define the observation function
    y = np.array([
        x[0]
    ])
    return y

# Define the initial state and covariance matrix
x_init = np.array([
    1.0,
    0.0
])
P_init = np.eye(len(x_init)) * 0.1

# Define the process noise and measurement noise covariance matrices
Q = np.eye(len(x_init)) * 0.01
R = np.eye(1) * 0.1

# Define the Jacobian matrices for f and h
def df_dx(x, u):
    # Jacobian of f w.r.t. x
    dfdx = np.array([
        [0, 1],
        [-2*mu*x[0]*x[1] - 1, mu*(1 - x[0]**2)]
    ])
    return dfdx

def df_du(x, u):
    # Jacobian of f w.r.t. u
    dfdu = np.array([
        [0],
        [1]
    ])
    return dfdu

def dh_dx(x):
    # Jacobian of h w.r.t. x
    dhdx = np.array([
        [1, 0]
    ])
    return dhdx

# Define the initial estimate and covariance matrix for the state
x_hat_init = np.array([
    0.5,
    -0.5
])
P_hat_init = np.eye(len(x_hat_init)) * 0.2

# Define the extended Kalman filter algorithm
def extended_kalman_filter(u, y):
    # Predict the next state estimate and covariance matrix using the nonlinear model and Jacobian matrices
    x_pred = f(x_hat, u)
    P_pred = df_dx(x_hat, u) @ P_hat @ df_dx(x_hat, u).T + Q
    
    # Update the state estimate and covariance matrix using the measurement and Jacobian matrices
    K = P_pred @ dh_dx(x_pred).T @ np.linalg.inv(dh_dx(x_pred) @ P_pred @ dh_dx(x_pred).T + R)
    x_hat = x_pred + K @ (y - h(x_pred))
    P_hat = (np.eye(len(x_hat)) - K @ dh_dx(x_pred)) @ P_pred
    
    return x_hat, P_hat

# Initialize the state estimate and covariance matrix
x_hat = x_hat_init
P_hat = P_hat_init

# Loop over the time steps and use the extended Kalman filter algorithm to estimate the state
num_time_steps = 100
time_steps = np.arange(num_time_steps)
u = np.sin(2*np.pi*time_steps/num_time_steps)
y_true = np.zeros((num_time_steps, 1))
y_meas = np.zeros((num_time_steps, 1))
x_true = x_init # Define the initial true state
for t in range(num_time_steps):
    # Get the control input and true state
    u_t = u[t]
    x_true_t = f(x_true, u_t)
    y_true_t = h(x_true_t)
    y_meas_t = y_true

    y_meas_t = y_true_t + np.random.normal(scale=np.sqrt(R[0,0]))
    y_true[t] = y_true_t
    y_meas[t] = y_meas_t

    # Use the extended Kalman filter algorithm to estimate the state
    x_hat, P_hat = extended_kalman_filter(u_t, y_meas_t)

    # Update the true state
    x_true = x_true_t

import matplotlib.pyplot as plt

plt.plot(time_steps, y_true, label='True state')
plt.plot(time_steps, y_meas, label='Measured state')
plt.plot(time_steps, x_hat[:,0], label='Estimated state')
plt.legend()
plt.show()
