"""
Functions that are helpful for sampling the parameters of a neutron star model.
There are some samplers we have built from the ground-up (some of which rely on
using models defined in `sm_models.py` and their Fisher matrix methods). If you
want to use `Bilby` then use the `KalmanLikelihood` defined below. See
`https://lscsoft.docs.ligo.org/bilby/index.html` for uses.

There are some example notebooks in this repository to help guide users as
well.
"""
import numpy as np
import bilby

class KalmanLikelihoodSecondSpindownGN(bilby.Likelihood):
    """
    Likelihood class for Bilby.

    In this case the `params` that are going to be passed around
    are exclusively dictionaries and are synonymous with the parameters
    in bilby.

    The parameters in the `param_map` function for the
    `neutron_star_model` used here *must* match the parameters passed as a
    keyword argument (or must match the default parameters shown in the
    __init__ script.
    """

    def __init__(self, neutron_star_model, parameters=None, x0=None):
        """
        Provides likelihood class for bilby. Must supply a maximum likelihood
        model (i.e. one of the
        """
        if parameters is None:
            parameters={'Qgamma': None, 'N2':None, 'beta': None,'delta':None,
                    'sigma':None, 'omg0':None, 'omgdot0': None,
                    'EFAC':None, 'EQUAD':None}
        super().__init__(parameters=parameters)
        self.neutron_star_model = neutron_star_model
        self.x0 = x0

    def log_likelihood(self):
        try:
            ll = self.neutron_star_model.loglike(self.parameters,
                    loglikelihood_burn=1)
        # if we can't run the kalman filter
        # these parameters are not right...
        except np.linalg.LinAlgError:
            ll= -np.inf
        # check if it's a nan.
        # if so don't use these parameters
        if np.isnan(ll):
            ll = -np.inf
        return ll

class KalmanLikelihoodSecondSpindown(bilby.Likelihood):
    """
    Likelihood class for Bilby.

    In this case the `params` that are going to be passed around
    are exclusively dictionaries and are synonymous with the parameters
    in bilby.

    The parameters in the `param_map` function for the
    `neutron_star_model` used here *must* match the parameters passed as a
    keyword argument (or must match the default parameters shown in the
    __init__ script.
    """

    def __init__(self, neutron_star_model, parameters=None, x0=None):
        """
        Provides likelihood class for bilby. Must supply a maximum likelihood
        model (i.e. one of the
        """
        if parameters is None:
            parameters={'Q1': None, 'Q2':None, 'N2': None, 'omg0':None, 'omgdot0': None,
                    'EFAC':None, 'EQUAD':None}
        super().__init__(parameters=parameters)
        self.neutron_star_model = neutron_star_model
        self.x0 = x0

    def log_likelihood(self):
        try:
            ll = self.neutron_star_model.loglike(self.parameters,
                    loglikelihood_burn=1)
        # if we can't run the kalman filter
        # these parameters are not right...
        except np.linalg.LinAlgError:
            ll= -np.inf
        # check if it's a nan.
        # if so don't use these parameters
        if np.isnan(ll):
            ll = -np.inf
        return ll

class KalmanLikelihoodOneComponent(bilby.Likelihood):
    """
    Likelihood class for Bilby.

    In this case the `params` that are going to be passed around
    are exclusively dictionaries and are synonymous with the parameters
    in bilby.

    The parameters in the `param_map` function for the
    `neutron_star_model` used here *must* match the parameters passed as a
    keyword argument (or must match the default parameters shown in the
    __init__ script.
    """

    def __init__(self, neutron_star_model, parameters=None, x0=None):
        """
        Provides likelihood class for bilby. Must supply a maximum likelihood
        model (i.e. one of the
        """
        if parameters is None:
            parameters={'sigma2': None, 'torque': None, 'omgc_0':None, 'EFAC':
                    None,
                    'EQUAD': None}
        super().__init__(parameters=parameters)
        self.neutron_star_model = neutron_star_model
        self.x0 = x0

    def log_likelihood(self):
        try:
            ll = self.neutron_star_model.loglike(self.parameters,
                    loglikelihood_burn=1)
        # if we can't run the kalman filter
        # these parameters are not right...
        except np.linalg.LinAlgError:
            ll= -np.inf
        # check if it's a nan.
        # if so don't use these parameters
        if np.isnan(ll):
            ll = -np.inf
        return ll

class KalmanLikelihood(bilby.Likelihood):
    """
    Likelihood class for Bilby.

    In this case the `params` that are going to be passed around
    are exclusively dictionaries and are synonymous with the parameters
    in bilby.

    The parameters in the `param_map` function for the
    `neutron_star_model` used here *must* match the parameters passed as a
    keyword argument (or must match the default parameters shown in the
    __init__ script.
    """

    def __init__(self, neutron_star_model, parameters=None, x0=None):
        """
        Provides likelihood class for bilby. Must supply a maximum likelihood
        model (i.e. one of the
        """
        if parameters is None:
            parameters={'relax_ratio': None,
                        'reduced_relax': None,
                        'Qc': None, 'Qs': None, 'lag': None,
                        'omegac_dot': None, 'omgc_0':None,
                        'omgs_0':None, 'EFAC': None,
                        'EQUAD': None}
        super().__init__(parameters=parameters)
        self.neutron_star_model = neutron_star_model
        self.x0 = x0

    def log_likelihood(self):
        try:
            ll = self.neutron_star_model.loglike(self.parameters,
                    loglikelihood_burn=1)
        # if we can't run the kalman filter
        # these parameters are not right...
        except np.linalg.LinAlgError:
            ll= -np.inf
        # check if it's a nan.
        # if so don't use these parameters
        if np.isnan(ll):
            ll = -np.inf
#         print(ll)
        return ll

class KalmanLikelihoodMeanReversion(bilby.Likelihood):
    """
    Likelihood class for Bilby.

    In this case the `params` that are going to be passed around
    are exclusively dictionaries and are synonymous with the parameters
    in bilby.

    The parameters in the `param_map` function for the
    `neutron_star_model` used here *must* match the parameters passed as a
    keyword argument (or must match the default parameters shown in the
    __init__ script.
    """

    def __init__(self, neutron_star_model, parameters=None, x0=None):
        """
        Provides likelihood class for bilby. Must supply a maximum likelihood
        model (i.e. one of the
        """
        if parameters is None:
            parameters={'sigma_v': None, 'sigma_a':None, 'N': None,'abar':None,
                    'gamma_v':None, 'gamma_a':None, 'fdot_start': None,
                    'fddot_start': None,
                    'EFAC':None, 'EQUAD':None}
        super().__init__(parameters=parameters)
        self.neutron_star_model = neutron_star_model
        self.x0 = x0

    def log_likelihood(self):
        try:
            ll = self.neutron_star_model.loglike(self.parameters,
                    loglikelihood_burn=1)
        # if we can't run the kalman filter
        # these parameters are not right...
        except np.linalg.LinAlgError:
            ll= -np.inf
        # check if it's a nan.
        # if so don't use these parameters
        if np.isnan(ll):
            ll = -np.inf
        return ll


