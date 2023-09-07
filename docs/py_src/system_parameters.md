# SystemParameters

[StateSpacePTA Index](../README.md#statespacepta-index) /
[Py Src](./index.md#py-src) /
SystemParameters

> Auto-generated documentation for [py_src.system_parameters](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/system_parameters.py) module.

- [SystemParameters](#systemparameters)
  - [SystemParameters](#systemparameters-1)

## SystemParameters

[Show source in system_parameters.py:12](https://github.com/tomkimpson/StateSpacePTA.jl/blob/pulsar_terms/py_src/system_parameters.py#L12)

#### Signature

```python
class SystemParameters:
    def __init__(
        self,
        NF=np.float64,
        T=10,
        cadence=7,
        Ω=5e-07,
        Φ0=0.2,
        ψ=2.5,
        ι=1.0,
        δ=1.0,
        α=1.0,
        h=0.01,
        σp=1e-13,
        σm=1e-08,
        Npsr=0,
        use_psr_terms_in_data=True,
        measurement_model="pulsar",
        seed=1234,
        σp_seed=1234,
        orthogonal_pulsars=False,
    ):
        ...
```