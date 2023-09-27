# import numpy as np
import numpy as np
# from numpy.linalg import inv, det, solve, cholesky,slogdet
from numpy.linalg import inv, det, solve, cholesky, slogdet
from numba import jit
from scipy.linalg import cho_solve


"""
Code for kalman filtering
and for using EM to solve for various
matrix elements.
"""

@jit(nopython=True)
def predict(x, P, transition, torque, process_covariance, nstates):
    xp = transition.dot(x) + torque
    Pp = transition @ P @ transition.T + process_covariance
    return xp, Pp

@jit(nopython=True)
def update(xp, Pp, measurement, emission, measurement_covariance, nmeasurements, nstates):
    err = measurement - emission @ xp
    inv_inn_cov = inv(emission @ Pp @ emission.T + measurement_covariance)
    gain = Pp @ emission.T @ inv_inn_cov
    x = xp + gain @ err
    P = (np.eye(nstates) - gain @ emission) @ (Pp)
    ll = -0.5 * (np.log(1/det(inv_inn_cov)) + err.T @ inv_inn_cov @ err + nmeasurements * np.log(2*np.pi))
    return x, P, ll

@jit(nopython=True)
def update_solve(xp, Pp, measurement, emission, measurement_covariance, nmeasurements, nstates):
    err = measurement - emission @ xp
    inn_cov = emission @ Pp @ emission.T + measurement_covariance
    gain = solve(inn_cov.T, (Pp @ emission.T).T).T
    x = xp + gain @ err
    P = (np.eye(nstates) - gain @ emission) @ (Pp)
    ll = -0.5 * (slogdet(inn_cov)[1] + err.T @ solve(inn_cov, err) + nmeasurements * np.log(2*np.pi))
    return x, P, ll

# TODO don't assume that covariances are diagonal. They're not given the
# transition matrix...

class KalmanFilterTimeVarying(object):
    """
    Kalman filter with time-varyaing matrices
    """
    def __init__(self, transition, emission, Q, R, B, solve=True):
        self.transition = transition  # Phi
        self.emission = emission  # H
        # infer number of state variables from transition size
        # infer number of measurement variables from emission size
        self.nmeasurements, self.nstates = np.shape(self.emission)
        # if these are square, take out the diagonal
        self.Q = Q  # nmeasurements x nstates x ntimes
        self.R = R  # nmeasurements x nmeasurements x ntimes
        self.B = B # nstates x ntimes
        # self.check_dimensions()
        self.ll = 0
        self.solve=solve

    def predict(self, timestep):
        self.xp, self.Pp = predict(self.x.reshape((self.nstates, 1)), self.P, self.transition[:, :, timestep], self.B[:, timestep].reshape((self.nstates, 1)),
                                   self.Q[:, :, timestep], self.nstates)

    def update(self, measurement, timestep):
        if self.solve:
            self.x, self.P, ll = update_solve(self.xp.reshape((self.nstates, 1)), self.Pp, measurement.reshape((self.nmeasurements, 1)), self.emission, self.R[:, :, timestep], self.nmeasurements, self.nstates)
        else:
            self.x, self.P, ll = update(self.xp.reshape((self.nstates, 1)), self.Pp, measurement.reshape((self.nmeasurements, 1)), self.emission, self.R[:, :, timestep], self.nmeasurements, self.nstates)
        self.ll += ll.squeeze()

    def ll_on_data(self, data, params=None, x0=None, P0=None, burn=1, return_states=False):
        # update parameters
        self.update_parameters(params)
        self.ll = 0
        self.xp = x0
        self.Pp = P0
        Nobs, Ndim = np.shape(data)
        if return_states:
            xx = np.zeros((Nobs, self.nstates))
            px = np.zeros((Nobs, self.nstates))
            lls = np.zeros(Nobs)
        self.update(data[0, :], 0)
        if return_states:
            xx[0, :] = self.x.squeeze()
            px[0, :] = np.diag(self.P)
            lls[0] = self.ll
        for nn in range(1, Nobs):
            if nn == burn:
                self.ll = 0
            self.predict(nn)
            self.update(data[nn, :], nn)
            if return_states:
                xx[nn, :] = self.x.squeeze()
                px[nn, :] = np.diag(self.P)
                lls[nn] = self.ll

        if return_states:
            return self.ll, xx, px, lls
        else:
            return self.ll

    def run_smoother(self, data, params=None, x0=None, P0=None, burn=1):
        ll, xx, px, lls = self.ll_on_data(data, params, x0, P0, burn, True)
        self.update_parameters(params)
        self.ll = 0
        self.xp = x0
        self.Pp = P0
        Nobs, Ndim = np.shape(data)
        xx = np.zeros((Nobs, self.nstates))
        px = np.zeros((Nobs, self.nstates))
        xP = np.zeros((Nobs, self.nstates))
        pP = np.zeros((Nobs, self.nstates))
        xS = np.zeros((Nobs, self.nstates))
        pS = np.zeros((Nobs, self.nstates))
        self.update(data[0, :], 0)
        xx[0, :] = self.x.squeeze()
        px[0, :] = np.diag(self.P)
        xP[0, :] = self.xp.squeeze()
        pP[0, :] = np.diag(self.Pp)
        lls[0] = self.ll
        for nn in range(1, Nobs):
            if nn == burn:
                self.ll = 0
            # predict
            self.predict(nn)
            # update
            self.update(data[nn, :], nn)
            # keep track (for smoother)
            xx[nn, :] = self.x.squeeze()
            px[nn, :] = np.diag(self.P)
            xP[nn, :] = self.xp.squeeze()
            pP[nn, :] = np.diag(self.Pp)
            lls[nn] = self.ll
        xS[-1, :] = self.x.squeeze()
        pS[-1, :] = np.diag(self.P)
        for nn in range(1, Nobs-1)[::-1]:
            C = np.diag(px[nn, :]) @ self.transition[:, :, nn+1].T @ inv(np.diag(pP[nn+1, :]))
            # print(C)
            # print(xx[nn, :] + C @ (xS[nn+1, :] - xP[nn+1, :]))

            xS[nn, :] = xx[nn, :] + C @ (xS[nn+1, :] - xP[nn+1, :])
            pS[nn, :] = np.diag(px[nn, :] + C @ (np.diag(pS[nn+1, :]) - np.diag(pP[nn+1, :])) @ C.T)

        return xS, pS


    def update_parameters(self, params):
        pass

class KalmanFilterTimeVaryingOneState(object):
    """
    Kalman filter with time-varyaing matrices
    """
    def __init__(self, transition, emission, Q, R, B):
        self.transition = transition  # Phi
        self.emission = emission  # H
        # infer number of state variables from transition size
        # infer number of measurement variables from emission size
        self.nmeasurements, self.nstates = np.shape(self.emission)
        # if these are square, take out the diagonal
        self.Q = Q  # nmeasurements x nstates x ntimes
        self.R = R  # nmeasurements x nmeasurements x ntimes
        self.B = B  # nstates x ntimes
        # self.check_dimensions()
        self.ll = 0

    def predict(self, timestep):
        #def predict(x, P, transition, torque, process_covariance, nstates):
        self.xp, self.Pp = predict(self.x.reshape((self.nstates, 1)),
                                   self.P,
                                   self.transition[timestep],
                                   self.B[timestep].reshape((self.nstates, 1)),
                                   self.Q[timestep],
                                   self.nstates)

    def update(self, measurement, timestep):
        #def update(xp, Pp, measurement, emission, measurement_covariance, nmeasurements, nstates):
        self.x, self.P, ll = update(self.xp.reshape((self.nstates, 1)),
                                    self.Pp,
                                    measurement.reshape((self.nmeasurements, 1)),
                                    self.emission, self.R[timestep].reshape(1,1),
                                    self.nmeasurements,
                                    self.nstates)
        self.ll += ll.squeeze()

    def ll_on_data(self, data, params=None, x0=None, P0=None, burn=1, return_states=False):
        # update parameters
        self.update_parameters(params)
        self.ll = 0
        self.xp = x0
        self.Pp = P0
        Nobs, Ndim = np.shape(data)
        if return_states:
            xx = np.zeros((Nobs, self.nstates))
            px = np.zeros((Nobs, self.nstates))
            lls = np.zeros(Nobs)
        self.update(data[0, :], 0)
        if return_states:
            xx[0, :] = self.x.squeeze()
            px[0, :] = np.diag(self.P)
            lls[0] = self.ll
        for nn in range(1, Nobs):
            if nn == burn:
                self.ll = 0
            self.predict(nn)
            self.update(data[nn, :], nn)
            if return_states:
                xx[nn, :] = self.x.squeeze()
                px[nn, :] = np.diag(self.P)
                lls[nn] = self.ll
        if return_states:
            return self.ll, xx, px, lls
        else:
            return self.ll

    def update_parameters(self, params):
        pass

