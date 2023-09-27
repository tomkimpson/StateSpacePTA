# baboo
A python package for doing fast neutron star inference using kalman filtering. Named after a cat. Work described in **paper link**.


# Installation

For now, this package can be installed by cloning and installing. I am hoping to add it to PyPI soon.

```bash
git clone https://github.com/meyers-academic/baboo.git
cd baboo
pip install .
```

# Current requirements

* `numpy`, `scipy`, `bilby`, `matplotlib`, `numba`
* For simulating arrival times and running on real data: `libstempo` and `TEMPO2` are required so that we cna generate a list of frequencies from a set of arrival times.

# Example notebooks

There are some example notebooks in the `example_notebooks` directory. Below we offer some highlights of what the package can do right now.

# Usage

## Simulation

* There is code for simulating frequencies directly from two-component model in `baboo/simulation.py`, which can be accessed by importing `baboo.simulation.two_component_fake_data`.

* There is code for simulating times of arrival (ToAs) for the two-component or one-component model. This has a slightly different api from the frequency code above, in part because the simulation code is more involved for generating ToAs. This code is accessed by creating a `baboo.simulation.TwoComponentModelSim` class (see `baboo.simulation.TwoComponentModelSim?` for details). Note that this code is for the same model as the `two_component_fake_data` function above, but takes a slightly different set of input parameters (there is a one-to-one mapping between the different sets of parameters, described in XXXX). Here is a quick example of how to simulate ToA, and then use `libstempo` to fit successive ToAs and generate a list of time-ordered frequencies.

```python
from baboo.simulation import TwoComponentModelSim

# list of rough times at which we will evaluate TOAs
# one toa per day (units of days)
tstarts = np.arange(500) 
toa_errors = np.ones(tstarts.size) * 1e-4 # 100 microsecond errors
lag = -1.6e-6  # lag between components
p0 = [0, 10, 10 - lag]  # inital states for our stochastic model

# instantiate model
mymodel = TwoComponentModelSim(r=3, tau=7.5e5, omgc_dot = -2.51e-12, lag=lag, Qc=1e-24, Qs=1e-24)

# Create ToAs, and then fit 3 ToAs at a time to generate a list of
# time-ordered frequencies using `libstempo`. `p0` is the starting states
# for the stochastic integrator. This depends on the model you are simulating.
# For this model it is [crust_phase, crust freq, superfluid freq] all evaluated 
# at the first entry of `tstart`. 
# 
# After it generates ToAs, this code fits N_toas_per_fit at a time to generate
# a list of time-ordered frequencies
# To do that, we need to tell it where to start for the fits,
# which are the second, third, and fourth arguments.
# 
# outputs are a list of toas, their errors,
# the times at which frequency measurements are made (times_fit),
# the frequency measurements from `libstempo`, and their errors.
# The last output is a list of the "true" state variables of the system
# (excluding the phase, which we get from ToAs), evaluated at
# the times of the toas.
# These can be compared to `freqs` as a good diagnostic to check whether
# fitting with `libstempo` is working properly and whether it is an accurate
# representation of the true frequency at that time.

toas, toa_errors, times_fit, freqs, freqs_errs, states = \
    mymodel.integrate_and_return_frequencies(tstarts, p0[1], omgc_dot,
                                             tstarts[0], toa_errors=toa_errors,
                                             p0=p0, Ntoas_per_fit=3)
```

At the moment, there are methods for simulating frequencies or ToAs for the two component, and for a much simpler one-component model. More documentation to come.

## Filtering 

We can run a Kalman filter to track the frequencies of the crust and the superfluid of the star using models in the `baboo.models` class. We create a model class by supplying parameters for our model and then we can run the filter on a set of time-ordered frequencies (and their uncertanties).  An example is like this:

```python
from baboo.models import TwoComponentModel


# Sometimes `libstempo` outputs np.longdouble (float128) values,
# but we want to use regular float64s.
# We also want to reshape the array so it is a proper column vector.
freqs = np.array(freqs).astype(float).reshape((np.size(times_fit), 1))

# The times that came out above for the frequencies are in MJD
# not seconds. So convert them. Also convert to regular float.
times_fit = 86400 * times_fit.astype(float)

# The design matrix relates the measurements to hidden states.
# measurements = design_matrix @ state_variables.
# In this case we have measurements of neutron star crust, 
# but not the superfluid. However, we still want to track the superfluid
# with the Kalman filter. So we have 1 set of measurements and 2 sets of states.
# So our design matrix is a 1x2 matrix.
design_matrix = np.array([1, 0]).reshape((1,2))

# Covariance matrix of measurements. In this case this comes from the errors
# of our fits above
mycov = np.asarray([np.eye(1) * freq_err for freq_err in freq_errs]).T

# Create a model object.
two_component_model = TwoComponentModel(freqs, mycov,
                                        design_matrix, times_fit)

# Run Kalman filter with a specific set of parameters
params = {'relax_ratio': 3, 'reduced_relax': 7.5e5**-1,
          'lag': lag, 'omegac_dot': -2.51e-12, 'Qc': 1e-24,
          'Qs': 1e-24, 'EFAC': 1, 'EQUAD': 0}

# Run filtering through calculating the log likelihood, but also 
# requst to # get back states. ll_total is the total log likelihood,
# state_estimates are the estimates of the crust and superfluid
# frequencies at each time step state_covariances are the
# covariance of those state estimates and ll_per_measurement is
# the log-likelihood assocated with each individual measurement.
ll_total, state_estimates, state_covariances, ll_per_measurement = two_component_model.loglike(params, return_states=True)

# If you are more interested in the state estimates, as opposed to the likelihood
# then for some models you can also run the `smooth` method, which runs a
# forward-backward algorithm to get better state estimates.
state_estimates, state_covariances = two_component_model.smooth(params)
```

## Sampling to find best model parameters give a set of data

What if we don't know the best parameters of the model beforehand (which is usually the case)? The Kalman filter has an associated likelihood. We can run the filter with different choices of model parameters and maximize the likelihood. In [this example](https://github.com/meyers-academic/baboo/blob/main/notebook_examples/baboo_two_component_example_sim_frequencies.ipynb) we show how to do this using `bilby` to perform MCMC sampling. I leave it to the notebook for that explanation.
