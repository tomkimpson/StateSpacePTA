# Bilby Wrapper

[StateSpacePta Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Bilby Wrapper

> Auto-generated documentation for [py_src.bilby_wrapper](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/bilby_wrapper.py) module.

- [Bilby Wrapper](#bilby-wrapper)
  - [BilbyLikelihood](#bilbylikelihood)
    - [BilbyLikelihood().log_likelihood](#bilbylikelihood()log_likelihood)
  - [BilbySampler](#bilbysampler)

## BilbyLikelihood

[Show source in bilby_wrapper.py:4](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/bilby_wrapper.py#L4)

#### Signature

```python
class BilbyLikelihood(bilby.Likelihood):
    def __init__(self, KalmanModel, parameters):
        ...
```

### BilbyLikelihood().log_likelihood

[Show source in bilby_wrapper.py:11](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/bilby_wrapper.py#L11)

#### Signature

```python
def log_likelihood(self):
    ...
```



## BilbySampler

[Show source in bilby_wrapper.py:19](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/bilby_wrapper.py#L19)

#### Signature

```python
def BilbySampler(KalmanFilter, init_parameters, priors, label, outdir):
    ...
```