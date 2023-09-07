# KalmanFilter

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
KalmanFilter

> Auto-generated documentation for [py_src.kalman_filter](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py) module.

- [KalmanFilter](#kalmanfilter)
  - [KalmanFilter](#kalmanfilter-1)
    - [KalmanFilter().likelihood](#kalmanfilter()likelihood)
  - [cauchy_likelihood](#cauchy_likelihood)
  - [log_likelihood](#log_likelihood)
  - [map_dicts_to_vector](#map_dicts_to_vector)
  - [predict](#predict)
  - [update](#update)

## KalmanFilter

[Show source in kalman_filter.py:82](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L82)

A class to implement the Kalman filter.

It takes two initialisation arguments:

`Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc.

`Observations`: class which holds the noisy observations recorded at the detector

#### Signature

```python
class KalmanFilter:
    def __init__(self, Model, Observations, PTA):
        ...
```

### KalmanFilter().likelihood

[Show source in kalman_filter.py:115](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L115)

#### Signature

```python
def likelihood(self, parameters):
    ...
```



## cauchy_likelihood

[Show source in kalman_filter.py:27](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L27)

#### Signature

```python
@jit(nopython=True)
def cauchy_likelihood(S, innovation):
    ...
```



## log_likelihood

[Show source in kalman_filter.py:16](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L16)

#### Signature

```python
@jit(nopython=True)
def log_likelihood(S, innovation):
    ...
```



## map_dicts_to_vector

[Show source in kalman_filter.py:202](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L202)

#### Signature

```python
def map_dicts_to_vector(parameters_dict):
    ...
```



## predict

[Show source in kalman_filter.py:73](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L73)

#### Signature

```python
@jit(nopython=True)
def predict(x, P, F, Q):
    ...
```



## update

[Show source in kalman_filter.py:42](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/kalman_filter.py#L42)

#### Signature

```python
@jit(nopython=True)
def update(x, P, observation, R, Xfactor, ephemeris):
    ...
```