# Priors

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Priors

> Auto-generated documentation for [py_src.priors](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py) module.

- [Priors](#priors)
  - [add_to_bibly_priors_dict_constant](#add_to_bibly_priors_dict_constant)
  - [add_to_bibly_priors_dict_log](#add_to_bibly_priors_dict_log)
  - [add_to_bibly_priors_dict_uniform](#add_to_bibly_priors_dict_uniform)
  - [add_to_priors_dict](#add_to_priors_dict)
  - [bilby_priors_dict](#bilby_priors_dict)
  - [priors_dict](#priors_dict)
  - [set_prior_on_measurement_parameters](#set_prior_on_measurement_parameters)
  - [set_prior_on_state_parameters](#set_prior_on_state_parameters)

## add_to_bibly_priors_dict_constant

[Show source in priors.py:14](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L14)

#### Signature

```python
def add_to_bibly_priors_dict_constant(x, label, init_parameters, priors):
    ...
```



## add_to_bibly_priors_dict_log

[Show source in priors.py:32](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L32)

#### Signature

```python
def add_to_bibly_priors_dict_log(x, label, init_parameters, priors, lower, upper):
    ...
```



## add_to_bibly_priors_dict_uniform

[Show source in priors.py:50](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L50)

#### Signature

```python
def add_to_bibly_priors_dict_uniform(x, label, init_parameters, priors, tol):
    ...
```



## add_to_priors_dict

[Show source in priors.py:68](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L68)

#### Signature

```python
def add_to_priors_dict(x, label, dict_A):
    ...
```



## bilby_priors_dict

[Show source in priors.py:204](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L204)

#### Signature

```python
def bilby_priors_dict(PTA, P):
    ...
```



## priors_dict

[Show source in priors.py:84](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L84)

#### Signature

```python
def priors_dict(pulsar_parameters, P):
    ...
```



## set_prior_on_measurement_parameters

[Show source in priors.py:132](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L132)

#### Signature

```python
def set_prior_on_measurement_parameters(init_parameters, priors, measurement_model, P):
    ...
```



## set_prior_on_state_parameters

[Show source in priors.py:107](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/priors.py#L107)

#### Signature

```python
def set_prior_on_state_parameters(init_parameters, priors, f, fdot, σp, γ, d):
    ...
```