# Main

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Main

> Auto-generated documentation for [py_src.main](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/main.py) module.

#### Attributes

- `P` - Setup the system: `SystemParameters(h=h, σp=None, σm=1e-11, use_psr_terms_in_data=True, measurement_model=measurement_model, seed=seed)`

- `model` - Define the model: `LinearModel(P)`

- `KF` - Initialise the Kalman filter: `KalmanFilter(model, data.f_measured, PTA)`

- `optimal_parameters` - Run the KF once with the correct parameters.
  This allows JIT precompile: `priors_dict(PTA, P)`
- [Main](#main)
