# Gravitational Waves

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Gravitational Waves

> Auto-generated documentation for [py_src.gravitational_waves](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py) module.

- [Gravitational Waves](#gravitational-waves)
  - [gw_earth_terms](#gw_earth_terms)
  - [gw_prefactors](#gw_prefactors)
  - [gw_psr_terms](#gw_psr_terms)
  - [h_amplitudes](#h_amplitudes)
  - [null_model](#null_model)
  - [principal_axes](#principal_axes)

## gw_earth_terms

[Show source in gravitational_waves.py:37](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py#L37)

#### Signature

```python
@jit(nopython=True)
def gw_earth_terms(delta, alpha, psi, q, q_products, h, iota, omega, d, t, phi0):
    ...
```



## gw_prefactors

[Show source in gravitational_waves.py:5](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py#L5)

#### Signature

```python
@jit(nopython=True)
def gw_prefactors(delta, alpha, psi, q, q_products, h, iota, omega, d, t, phi0):
    ...
```



## gw_psr_terms

[Show source in gravitational_waves.py:49](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py#L49)

#### Signature

```python
@jit(nopython=True)
def gw_psr_terms(delta, alpha, psi, q, q_products, h, iota, omega, d, t, phi0):
    ...
```



## h_amplitudes

[Show source in gravitational_waves.py:93](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py#L93)

#### Signature

```python
@jit(nopython=True)
def h_amplitudes(h, ι):
    ...
```



## null_model

[Show source in gravitational_waves.py:71](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py#L71)

#### Signature

```python
@jit(nopython=True)
def null_model(delta, alpha, psi, q, q_products, h, iota, omega, d, t, phi0):
    ...
```



## principal_axes

[Show source in gravitational_waves.py:78](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/gravitational_waves.py#L78)

#### Signature

```python
@jit(nopython=True)
def principal_axes(theta, phi, psi):
    ...
```