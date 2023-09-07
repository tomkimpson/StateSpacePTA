# Model

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Model

> Auto-generated documentation for [py_src.model](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/model.py) module.

- [Model](#model)
  - [LinearModel](#linearmodel)
  - [F_function](#f_function)
  - [Q_function](#q_function)
  - [R_function](#r_function)

## LinearModel

[Show source in model.py:9](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/model.py#L9)

A linear model of the state evolution

#### Signature

```python
class LinearModel:
    def __init__(self, P):
        ...
```



## F_function

[Show source in model.py:40](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/model.py#L40)

#### Signature

```python
@jit(nopython=True)
def F_function(gamma, dt):
    ...
```



## Q_function

[Show source in model.py:47](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/model.py#L47)

#### Signature

```python
@jit(nopython=True)
def Q_function(gamma, sigma_p, dt):
    ...
```



## R_function

[Show source in model.py:55](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/model.py#L55)

#### Signature

```python
@jit(nopython=True)
def R_function(sigma_m):
    ...
```