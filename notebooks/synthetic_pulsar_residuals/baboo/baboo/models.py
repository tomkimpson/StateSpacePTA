"""
Models defined using Kalman Filter in `kalman.py`

List of Models:

    `TwoComponentModel`: Two component model using full solution and
    non-uniform time sampling. THIS IS THE ONE YOU PROBABLY WANT TO USE.

    `OneComponentModel`: Simple one-component model with spin-wandering.

    `SecondSpindownModel`: Include a second spindown and track frequency derivative.
"""
# import numpy as np
import numpy as np
from .kalman import KalmanFilterTimeVarying, KalmanFilterTimeVaryingOneState
import bilby

class TwoComponentModel(KalmanFilterTimeVarying):
    """
    Two component model class. Usually you will initialize this
    by specifying the measurements, times, measurement covariance,
    and design matrix. The other parameters transition matrix,
    torque matrix, and process noise covariance, depend on your specific
    model, and can be supplied when calling specific methods, like smoothing,
    or getting a likelihood.

    Parameters:
    -----------
    endog : np.ndarray
        measurements you have made. Ntimes x Nmeasurements.
    measurement_cov : np.ndarray
        covariance on measurements. Ntimes x Nmeasurements x Nmeasurements.
    design : np.ndarray
        Matrix linking states to measurements. Nmeasurements x Nstates
    times : np.ndarray
        Times at which measurements are made. Ntimes x 1
    Q : np.ndarray [not usually initially supplied]
        Process noise covariance matrix
    transition : np.ndarray [not usually initially supplied]
        Transition matrix
    B :  np.ndarray [not usually initially supplied]
        Torque matrix
    solve : bool
        State whether to explicitly caculate inverse matrices (False)
        or solve linear equation using `np.solve` (True) when running
        Kalman filter.
    params : dict, [not usually supplied on instantiation]
        parameters of the model
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None,
                 Q=None, transition=None, B=None, solve=True, params=None):
        super(TwoComponentModel, self).__init__(transition, design, Q, measurement_cov, B, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'relax_ratio': None, 'reduced_relax': None,
                      'lag': None, 'omegac_dot': None,
                      'Qc': None, 'Qs': None,
                      'EFAC': None,'EQUAD': None}
        else:
            self.params = params.copy()

    @property
    def transformed_params(self):
        A, B, C, D = self.params['relax_ratio'], (self.params['reduced_relax'])**-1, self.params['lag'], self.params['omegac_dot']
        taus = (1+A) * B
        tauc = (1+A) / A * B
        qc = self.params['Qc']
        qs = self.params['Qs']
        Nc = (D + A/(1+A) * C/B)
        Ns = D - C/B * (1+A)**-1
        return tauc, taus, qc, qs, Nc, Ns, self.params['EFAC'], self.params['EQUAD']

    @property
    def keylist(self):
        return ['relax_ratio', 'reduced_relax',
                   'lag', 'omegac_dot', 'Qc', 'Qs',
                   'EFAC','EQUAD']

    def _check_params(self, params):
        if list(self.params.keys()) != list(params.keys()):
            raise ValueError(f"Incorrect set of parameters supplied. List of available parameters is {self.keylist}")


    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # check
        self._check_params(params)
        # update
        self.params = params.copy()
        tauc, taus, Qc, Qs, Nc, Ns, EFAC, EQUAD = self.transformed_params
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition(tauc, taus, dts)
        self.update_Q(tauc, taus, Qc, Qs, dts)
        self.update_torque(tauc, taus, Nc, Ns, dts)
        # rescale and add more uncertainty, as done typically with pulsar # timing
        self.R[0, 0, :] = self._R[0, 0, :] * EFAC + EQUAD

    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewha reasonable starting point for
        # initial state. I dont' think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        try:
            self.params = params.copy()
            tauc, taus, Qc, Qs, Nc, Ns, EFAC, EQUAD = self.transformed_params
            lag = (tauc * taus / (tauc + taus)) * (Nc - Ns)
            if isinstance(params, dict):
                omgc_0 = params['omgc_0']
                omgs_0 = omgc_0 - lag
            elif isinstance(params, np.ndarray):
                omgc_0 = self.data[0,0]
                omgs_0 = omgc_0 - lag
            return self.ll_on_data(self.data, params, x0=np.array([omgc_0, omgs_0]),
                                   P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e1,
                                   burn=loglikelihood_burn,
                                   return_states=return_states)
        except np.linalg.LinAlgError:
            return -np.inf

    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        tauc, taus, Qc, Qs, Nc, Ns, EFAC, EQUAD = self.transformed_params
        lag = (tauc * taus / (tauc + taus)) * (Nc - Ns)
        if isinstance(params, dict):
            omgc_0 = params['omgc_0']
            omgs_0 = omgc_0 - lag
        elif isinstance(params, np.ndarray):
            omgc_0 = self.data[0,0]
            omgs_0 = omgc_0 - lag
        return self.run_smoother(self.data, params, x0=np.array([omgc_0, omgs_0]),
                               P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e5,
                               )


    def update_Q(self, tauc, taus, Qc, Qs, dts):
        tauc = np.real(tauc)
        taus = np.real(taus)
        Qc = np.real(Qc)
        Qs = np.real(Qs)
        dts = np.real(dts)
        # useful values
        const = (1/tauc + 1/taus)
        tauc_plus_taus_squared = (tauc + taus)**2
        expvals = np.exp(-const * dts).real
        Q = np.zeros((2, 2, dts.size))
        tau = const**-1
        Q[0,0,:] = (Qc * tauc**2 + Qs*taus**2) * dts +\
                 (2 * Qc *tauc * taus - 2 * Qs * taus**2) * tau * (1 - expvals) +\
                 (Qc * taus**2 + Qs * taus**2) * (1 - expvals**2) * (tau / 2)

        Q[1,1,:] = (Qc * tauc**2 + Qs*taus**2) * dts +\
                (2 * Qs *tauc * taus - 2 * Qc * tauc**2) * tau * (1 - expvals) +\
                (Qc * tauc**2 + Qs * tauc**2) * (1 - expvals**2) * (tau / 2)

        Q[1,0,:] = (Qs*taus**2 + Qc*tauc**2) * dts + \
                (Qc * tauc * taus - tauc**2*Qc + Qs*tauc*taus - Qs*taus**2)*tau*(1 - expvals) - \
                (Qc + Qs)*(tauc*taus)*(1-expvals**2)*(tau/2)
        Q[0,1,:]  = Q[1, 0, :].copy()
        self.Q =  Q / (tauc_plus_taus_squared)

            # exponential transition matrix
    def update_transition(self, tauc, taus, dts):
        tau = (tauc * taus) / (tauc + taus)
        expvals = np.exp(-dts / tau)
        transition = np.zeros((2,2,dts.size))
        transition[0, 0, :] = tauc + taus * expvals
        transition[0, 1, :] = taus - taus * expvals
        transition[1, 0, :] = tauc - tauc * expvals
        transition[1, 1, :] = taus + tauc * expvals
        self.transition = transition / (tauc + taus)

    def update_torque(self, tauc, taus, Nc, Ns, dts):
        tau = (tauc * taus) / (tauc + taus)
        tauc_plus_taus_squared = (tauc + taus)**2
        expvals = np.exp(-dts / tau)
        omg_dot = (tauc * Nc + taus * Ns) / (tauc + taus)
        torques = np.zeros((2, dts.size))

        torques[0, :] = omg_dot * dts + \
                        tau**2 * (tauc**-1 * (Nc - Ns) * (1 - expvals))
        torques[1, :] = omg_dot * dts + \
                        tau**2 * (taus**-1 * (Ns - Nc) * (1 - expvals))
        self.B = torques

class OneComponentModel(KalmanFilterTimeVaryingOneState):
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, Q=None,
                 transition=None, B=None):
        super(OneComponentModel, self).__init__(transition, design, Q,
                                                measurement_cov, B)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self._R = self.R.copy()

    def param_map(self, params):
        # Remember to adjust the param labels in the sample.py file
        Q, N = params['sigma2'], params['torque']
        return Q, N, params['EFAC'], params['EQUAD']

    @property
    def keylist(self):
        return ['sigma2', 'torque', 'omgc_0', 'EFAC', 'EQUAD']

    def _check_params(self, params):
        # print(list(params.keys()))
        if self.keylist != list(params.keys()):
            raise ValueError(f"Incorrect set of parameters supplied. List of available parameters is {self.keylist}")


    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        self._check_params(params)
        Q, N, EFAC, EQUAD = self.param_map(params)
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition_fast(dts)
        self.update_Q_fast(Q, dts)
        self.update_torques_fast(N, dts)
        # self.R = self.R * EFAC + EQUAD
        self.R = self._R * EFAC + EQUAD

    def update_transition_fast(self, dts):
        self.transition = transition.copy()
        transition = np.ones((dts.size, 1, 1))

    def update_Q_fast(self, Q, dts):
        Qs = np.zeros((dts.size, 1, 1))
        Qs[:,0, 0] = Q*dts
        self.Q = Qs.copy()

    def update_torques_fast(self, N, dts):
        torques = np.zeros((dts.size, 1, 1))
        torques[:,0, 0] = N*dts
        self.B = torques.copy()


    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        if self.param_map is None:
            # assume params are the three params we want.
            Q, N, EFAC, EQUAD = params
        else:
            Q, N, EFAC, EQUAD = self.param_map(params)
        if isinstance(params, dict):
            omgc_0 = params['omgc_0']
        elif isinstance(params, list):
            omgc_0 = params[-1]
        # print(Q, N, EFAC, EQUAD, omgc_0)
        x = self.ll_on_data(self.data, params, x0=np.array([omgc_0]),
                            P0=np.eye(self.nstates) * np.max(self.R)*1e1,
                            burn=loglikelihood_burn,
                            return_states=return_states)
        return x
    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information

        Q, N, EFAC, EQUAD = params
        if isinstance(params, dict):
            omgc_0 = params['omgc_0']
        elif isinstance(params, list):
            omgc_0 = params[-1]
        return self.run_smoother(self.data, params, x0=np.array([omgc_0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               )



class SecondSpindownModel(KalmanFilterTimeVarying):
    """
    note that param map function can't contain dependence on dt
    for this model.
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, Q=None,
                 transition=None, B=None, solve=True, params=None):
        super(SecondSpindownModel, self).__init__(transition, design, Q, measurement_cov, B, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'Q1': None, 'Q2':None, 'N2': None, 'EFAC': None, 'EQUAD': None}
        else:
            self.params = params


    @property
    def transformed_params(self):
        return self.params['Q1'],self.params['Q2'],self.params['N2'], self.params['EFAC'], self.params['EQUAD']

    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # update parameters
        self.params = params.copy()
        # get transformed version
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition(dts)
        self.update_Q(Q1, Q2, dts)
        self.update_torques(N2, dts)
        self.R = self._R * EFAC + EQUAD

    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        if isinstance(params, dict):
            omg0 = params['omg0']
            omgdot0 = params['omgdot0']
        elif isinstance(params, np.ndarray):
            omg0 = self.data[0,0]
        return self.ll_on_data(self.data, params, x0=np.array([omg0, omgdot0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               burn=loglikelihood_burn,
                               return_states=return_states)
    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        if isinstance(params, dict):
            omg0 = params['omg0']
            omgdot0 = params['omgdot0']
        elif isinstance(params, np.ndarray):
            omg0 = self.data[0,0]
        return self.run_smoother(self.data, params, x0=np.array([omg0, omgdot0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               )


    def update_transition(self, dts):
        transition = np.zeros((2, 2, dts.size))
        transition[0, 0, :] = 1
        transition[1, 1, :] = 1
        transition[0, 1, :] = dts
        self.transition = transition


    def update_Q(self, Q1, Q2, dts):
        Qs = np.zeros((2, 2, dts.size))
        Qs[0, 0, :] = Q1*dts+Q2*dts**3/3
        Qs[1, 1, :] = Q2*dts
        Qs[0, 1, :] = (Q2*dts**2)/2
        Qs[1, 0, :] = (Q2*dts**2)/2
        self.Q = Qs

    def update_torques(self, N, dts):
        torques = np.zeros((2, dts.size))
        torques[0, :] = N*dts**2 / 2
        torques[1, :] = N*dts
        self.B = torques

class SecondSpindownModelGeneralNoise(KalmanFilterTimeVarying):
    """
    note that param map function can't contain dependence on dt
    for this model.
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, Q=None,
                 transition=None, B=None, solve=True, params=None):
        super(SecondSpindownModelGeneralNoise, self).__init__(transition, design, Q, measurement_cov, B, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'Qgamma': None, 'N2': None, 'beta': None, 'delta': None, 'sigma': None, 'EFAC': None, 'EQUAD': None}
        else:
            self.params = params


    @property
    def transformed_params(self):
        return self.params['Qgamma'],self.params['N2'],self.params['beta'],self.params['delta'], self.params['sigma'], self.params['EFAC'], self.params['EQUAD']

    def update_parameters(self, params):
        """
        update transition matrix, etc.
        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # update parameters
        self.params = params.copy()
        # get transformed version
        Qgamma, N2, beta, delta, sigma, EFAC, EQUAD = self.transformed_params
        dts = self.times[1:] - self.times[:-1]
        dts = np.append(1, dts)
        # construct transitions, etc.
        self.update_transition(beta, delta, sigma, dts)
        self.update_Q(Qgamma, dts)
        self.update_torques(N2, dts)
        self.R = self._R * EFAC + EQUAD

    # make loglike more like statsmodel
    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        Qgamma, N2, beta, delta, sigma, EFAC, EQUAD = self.transformed_params
        # Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        if isinstance(params, dict):
            omg0 = params['omg0']
            omgdot0 = params['omgdot0']
        elif isinstance(params, np.ndarray):
            omg0 = self.data[0,0]
        return self.ll_on_data(self.data, params, x0=np.array([omg0, omgdot0, 0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               burn=loglikelihood_burn,
                               return_states=return_states)
    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        Q1, Q2, N2, EFAC, EQUAD = self.transformed_params
        if isinstance(params, dict):
            omg0 = params['omg0']
            omgdot0 = params['omgdot0']
        elif isinstance(params, np.ndarray):
            omg0 = self.data[0,0]
        return self.run_smoother(self.data, params, x0=np.array([omg0, omgdot0]),
                               P0=np.eye(self.nstates) * np.max(self.R.flatten())*1e5,
                               )


    def update_transition(self, beta, delta, sigma, dts):
        #TODO Andrés update
        transition = np.zeros((3, 3, dts.size))
        transition[0, 0, :] = 1
        transition[1, 1, :] = 1
        transition[0, 1, :] = dts
        transition[0, 2, :] = (beta*sigma + np.exp(-dts*sigma)*(delta-beta*sigma)-delta*(1-dts*sigma))/sigma**2
        transition[1, 2, :] = delta *(np.exp(-dts*sigma)-1)
        transition[2, 2, :] = np.exp(-dts*sigma)
        self.transition = transition

    def update_Q(self, Qgamma, dts):
        Qs = np.zeros((3, 3, dts.size))
        Qs[2, 2, :] = (Qgamma)*dts
        self.Q = Qs

    def update_torques(self, N, dts):
        torques = np.zeros((3, dts.size))
        torques[0, :] = N*dts**2 / 2
        torques[1, :] = N*dts
        self.B = torques


class MeanRevertingModel(KalmanFilterTimeVarying):
    """
    note that param map function can't contain dependence on dt
    for this model.
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, solve=True, params=None):
        # transition, Q, and B matrices are set to None for
        # now. They will be specified when parameters are given.
        super(MeanRevertingModel, self).__init__(None, design, None,
              measurement_cov, None, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'sigma_v': None, 'sigma_a': None,
                           'N': None, 'abar': None, 'gamma_v': None,
                           'gamma_a': None, 'EFAC': None, 'EQUAD': None}
        else:
            self.params = params

    @property
    def keylist(self):
        return ['sigma_v', 'sigma_a', 'N', 'abar', 'gamma_v', 'gamma_a',
         'EFAC', 'EQUAD', 'fdot_start', 'fddot_start']

    def _check_params(self, params):
        # print(list(params.keys()))
        # if self.keylist != list(params.keys()):
        #    raise ValueError(f"Incorrect set of parameters supplied. List of available parameters is {self.keylist}")
        pass

    @property
    def dts(self):
        return np.append(1, self.times[1:] - self.times[:-1])

    def update_parameters(self, params):
        self._check_params(params)
        self.update_transition(params['gamma_v'], params['gamma_a'])
        self.update_Q(params['sigma_v'], params['sigma_a'], params['gamma_v'],
                params['gamma_a'])
        self.update_torque(params['gamma_v'], params['gamma_a'], params['N'],
                params['abar'])

    def update_transition(self, gamma_v, gamma_a):
        transition = np.zeros((2, 2, self.dts.size))
        exp_gamma_a = np.exp(-gamma_a * self.dts)
        exp_gamma_v = np.exp(-gamma_v * self.dts)
        transition[0, 0, :] = exp_gamma_v
        transition[0, 1, :] = (exp_gamma_v - exp_gamma_a) / (gamma_a - gamma_v)
        transition[1, 1, :] = exp_gamma_a
        self.transition = transition

    def update_Q(self, sigma_v, sigma_a, gamma_v, gamma_a):
        Q = np.zeros((2, 2, self.dts.size))
        exp_gamma_a = np.exp(-gamma_a * self.dts)
        exp_gamma_v = np.exp(-gamma_v * self.dts)
        gamma_tot = gamma_a + gamma_v
        exp_gamma_tot = np.exp(-gamma_tot * self.dts)
        siga2 =  sigma_a**2
        sigv2 = sigma_v**2
        Q[0, 0, :] = (siga2 / gamma_a) * (1 - exp_gamma_a**2) \
                  - 4 * siga2 / gamma_tot \
                  + 4 * siga2 * exp_gamma_tot / (gamma_tot) \
                  + ((siga2 + (gamma_a - gamma_v)**2 * sigv2) / gamma_v) \
                  * (1 - exp_gamma_v**2)
        Q[0, 0, :] *= 0.5 * 1 / (gamma_a - gamma_v)**2
        Q[1, 0, :] = 0.5 * (exp_gamma_a**2 - 1) / (gamma_a * (gamma_a - gamma_v)) \
                     - (exp_gamma_tot - 1) / (gamma_a**2 - gamma_v**2)
        Q[1, 0, :] *= siga2
        Q[0, 1, :] = Q[1, 0, :]
        Q[1, 1, :] = (1 - exp_gamma_a**2) * siga2 / (2 * gamma_a)
        self.Q = Q

    def update_torque(self, gamma_v, gamma_a, N, abar):
        exp_gamma_a = np.exp(-gamma_a * self.dts)
        exp_gamma_v = np.exp(-gamma_v * self.dts)
        gamma_tot = gamma_a + gamma_v
        exp_gamma_tot = np.exp(-gamma_tot * self.dts)

        torque = np.zeros((2, self.dts.size))
        torque[0, :] = N * (1 - exp_gamma_v) + \
                  (abar * ((1 - exp_gamma_v) * gamma_a - (1 - exp_gamma_a) *
                      gamma_v) / ((gamma_a - gamma_v) * gamma_v))
        torque[1, :] = (1 - exp_gamma_a) * abar
        self.B = torque

    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewha reasonable starting point for
        # initial state. I dont' think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        # try:
        self.params = params.copy()
        if isinstance(params, dict):
            fdot_start = params['fdot_start']
            fddot_start = params['fddot_start']
        elif isinstance(params, np.ndarray):
            fdot_start = self.data[0,0]
            fddot_start = 0
        return self.ll_on_data(self.data, params, x0=np.array([fdot_start, fddot_start]),
                               P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e1,
                               burn=loglikelihood_burn,
                               return_states=return_states)
#         except np.linalg.LinAlgError:
#             return -np.inf
    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        self.params = params.copy()
        print(self.params)
        if isinstance(params, dict):
            fdot_start = params['fdot_start']
            fddot_start = params['fddot_start']
        elif isinstance(params, np.ndarray):
            fdot_start = self.data[0,0]
            fddot_start = 0
        return self.run_smoother(self.data, params, x0=np.array([fdot_start, fddot_start]),
                               P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e5,
                               )

class MeanRevertingModel2(KalmanFilterTimeVarying):
    """
    note that param map function can't contain dependence on dt
    for this model.
    """
    def __init__(self, endog, measurement_cov=None,
                 design=None, times=None, solve=True, params=None):
        # transition, Q, and B matrices are set to None for
        # now. They will be specified when parameters are given.
        super(MeanRevertingModel2, self).__init__(None, design, None,
              measurement_cov, None, solve)
        if times is None:
            raise ValueError("must specify variable dt")
        self.times = times
        self.data = endog
        self.nobs = self.times.size
        self.solve = solve
        self._R = self.R.copy()
        if params is None:
            self.params = {'sigma_v': None, 'sigma_a': None,
                           'N': None, 'abar': None, 'gamma_v': None,
                           'gamma_a': None, 'EFAC': None, 'EQUAD': None}
        else:
            self.params = params

    @property
    def keylist(self):
        return ['sigma_v', 'sigma_a', 'N', 'abar', 'gamma_v', 'gamma_a',
         'EFAC', 'EQUAD', 'fdot_start', 'fddot_start']

    def _check_params(self, params):
        # print(list(params.keys()))
        # if self.keylist != list(params.keys()):
        #    raise ValueError(f"Incorrect set of parameters supplied. List of available parameters is {self.keylist}")
        pass

    @property
    def dts(self):
        return np.append(1, self.times[1:] - self.times[:-1])

    def update_parameters(self, params):
        self._check_params(params)
        self.update_transition(params['gamma_v'], params['gamma_a'])
        self.update_Q(params['sigma_v'], params['sigma_a'], params['gamma_v'],
                params['gamma_a'])
        self.update_torque(params['gamma_v'], params['gamma_a'], params['N'],
                params['abar'])

    def update_transition(self, gamma_v, gamma_a):
        transition = np.zeros((2, 2, self.dts.size))
        exp_gamma_a = np.exp(-gamma_a * self.dts)
        transition[0, 0, :] = 1
        transition[0, 1, :] = (1 - exp_gamma_a) / gamma_a
        transition[1, 1, :] = exp_gamma_a
        self.transition = transition

    def update_Q(self, sigma_v, sigma_a, gamma_v, gamma_a):
        Q = np.zeros((2, 2, self.dts.size))
        exp_gamma_a = np.exp(-gamma_a * self.dts)
        exp_gamma_v = np.exp(-gamma_v * self.dts)
        gamma_tot = gamma_a + gamma_v
        exp_gamma_tot = np.exp(-gamma_tot * self.dts)
        siga2 =  sigma_a**2
        sigv2 = sigma_v**2
        Q[0, 0, :] = -(3 + exp_gamma_a**2 - 4 * exp_gamma_a - 2 * self.dts * gamma_a) * siga2 / (2 * gamma_a**3) + self.dts * sigv2
        Q[1, 0, :] = (1 - 2*exp_gamma_a + exp_gamma_a**2) * siga2 / (2 * gamma_a**2)
        Q[0, 1, :] = Q[1, 0, :]
        Q[1, 1, :] = (1 - exp_gamma_a**2) * siga2 / (2 * gamma_a)
        self.Q = Q

    def update_torque(self, gamma_v, gamma_a, N, abar):
        exp_gamma_a = np.exp(-gamma_a * self.dts)
        exp_gamma_v = np.exp(-gamma_v * self.dts)
        gamma_tot = gamma_a + gamma_v
        exp_gamma_tot = np.exp(-gamma_tot * self.dts)

        torque = np.zeros((2, self.dts.size))
        torque[0, :] = abar * (self.dts + (exp_gamma_a - 1) / gamma_a)
        torque[1, :] = (1 - exp_gamma_a) * abar
        self.B = torque

    def loglike(self, params, loglikelihood_burn=1, return_states=False):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewha reasonable starting point for
        # initial state. I dont' think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        # try:
        self.params = params.copy()
        if isinstance(params, dict):
            fdot_start = params['fdot_start']
            fddot_start = params['fddot_start']
        elif isinstance(params, np.ndarray):
            fdot_start = self.data[0,0]
            fddot_start = 0
        return self.ll_on_data(self.data, params, x0=np.array([fdot_start, fddot_start]),
                               P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e1,
                               burn=loglikelihood_burn,
                               return_states=return_states)
#         except np.linalg.LinAlgError:
#             return -np.inf
    def smooth(self, params):
        """
        calculate log likelihood on data for a given set of parameters.

        params is either a list or a dictionary that can be passed
        to `self.param_map`.
        """
        # start at somewhat reasonable starting point for
        # initial state. I don't think this is strictly correct.
        # we should figure out how to properly start this with
        # diffuse starting information
        self.params = params.copy()
        print(self.params)
        if isinstance(params, dict):
            fdot_start = params['fdot_start']
            fddot_start = params['fddot_start']
        elif isinstance(params, np.ndarray):
            fdot_start = self.data[0,0]
            fddot_start = 0
        return self.run_smoother(self.data, params, x0=np.array([fdot_start, fddot_start]),
                               P0=np.eye(self.nstates) * np.max(self.R[:, :, 0])*1e5,
                               )


