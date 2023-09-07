# Plotting

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
Plotting

> Auto-generated documentation for [py_src.plotting](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/plotting.py) module.

- [Plotting](#plotting)
  - [SNR_plots](#snr_plots)
  - [plot_all](#plot_all)
  - [plot_custom_corner](#plot_custom_corner)
  - [plot_likelihood](#plot_likelihood)
  - [plot_statespace](#plot_statespace)

## SNR_plots

[Show source in plotting.py:222](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/plotting.py#L222)

#### Signature

```python
def SNR_plots(x, y1, y2, xlabel, savefig=None):
    ...
```



## plot_all

[Show source in plotting.py:30](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/plotting.py#L30)

#### Signature

```python
def plot_all(
    t,
    states,
    measurements,
    measurements_clean,
    predictions_x,
    predictions_y,
    psr_index,
    savefig=None,
):
    ...
```



## plot_custom_corner

[Show source in plotting.py:101](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/plotting.py#L101)

#### Signature

```python
def plot_custom_corner(
    path,
    variables_to_plot,
    labels,
    injection_parameters,
    ranges,
    axes_scales,
    scalings=[1.0, 1.0],
    savefig=None,
    logscale=False,
    title=None,
    smooth=True,
    smooth1d=True,
):
    ...
```



## plot_likelihood

[Show source in plotting.py:192](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/plotting.py#L192)

#### Signature

```python
def plot_likelihood(x, y, parameter_name, log_x_axes=False):
    ...
```



## plot_statespace

[Show source in plotting.py:10](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/plotting.py#L10)

#### Signature

```python
def plot_statespace(t, states, measurements, psr_index):
    ...
```