# Pulsars

[StateSpacePta Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Pulsars

> Auto-generated documentation for [py_src.pulsars](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/pulsars.py) module.

- [Pulsars](#pulsars)
  - [Pulsars](#pulsars-1)
  - [convert_vector_to_ra_dec](#convert_vector_to_ra_dec)
  - [random_three_vector](#random_three_vector)
  - [unit_vector](#unit_vector)

## Pulsars

[Show source in pulsars.py:7](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/pulsars.py#L7)

#### Signature

```python
class Pulsars:
    def __init__(self, SystemParameters):
        ...
```



## convert_vector_to_ra_dec

[Show source in pulsars.py:118](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/pulsars.py#L118)

#### Signature

```python
def convert_vector_to_ra_dec(v):
    ...
```



## random_three_vector

[Show source in pulsars.py:102](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/pulsars.py#L102)

Generates a random 3D unit vector (direction) with a uniform spherical distribution
Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution

#### Signature

```python
def random_three_vector():
    ...
```



## unit_vector

[Show source in pulsars.py:93](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/pulsars.py#L93)

#### Signature

```python
def unit_vector(theta, phi):
    ...
```