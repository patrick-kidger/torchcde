# torchcontroldiffeq
This library provides differentiable GPU-capable solvers for integrating controlled differential equations, implemented in PyTorch.

Backpropagation via the adjoint method is supported and allows for improved memory efficiency.

In particular, this allows for building [Neural Controlled Differential Equation](https://github.com/patrick-kidger/NeuralCDE) models, which extend Neural ODE models to (irregular) time series.

_Powered by the excellent [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library._

## Installation

```bash
pip install git+https://github.com/patrick-kidger/torchcontroldiffeq.git
```

## Example
We encourage looking at [example.py](./example/example.py), which demonstrates how to use the library to train a Neural CDE model to predict the chirality of a spiral.

## Citation
If you found use this library useful, we would appreciate a citation:

```bibtex
@article{kidger2020neuralcde,
    author={Kidger, Patrick and Morrill, James and Foster, James and Lyons, Terry},
    title={{Neural Controlled Differential Equations for Irregular Time Series}},
    year={2020},
    journal={arXiv:2005.08926}
}
```

## Documentation

The library consists of two main components: (1) integrators for solving controlled differential equations, and (2) ways of constructing controls from data.

### Integrators

The library provides the `cdeint` function, which solves the system of controlled differential equations:
```
dz = f(z)dX     z(t_0) = z0
```

The goal is to find the response `z` driven by the control `X`. For our purposes here, this can be re-written as the following differential equation:
```
dz/dt = f(z)dX/dt     z(t_0) = z0
```
where the right hand side describes a matrix-vector product between `f(z)` and `dX/dt`.

This is solved by
```python
cdeint(X, func, z0, t, adjoint=True, **kwargs)
```
where letting `...` denote an arbitrary number of batch dimensions:
- `X.derivative(t)` is a Tensor of shape `(..., input_channels)`,
- `func(z)` is a Tensor of shape `(..., hidden_channels, input_channels)`,
- `z0` a Tensor of shape `(..., hidden_channels)`,
- `t` is a one-dimensional Tensor of times.

The adjoint method can be toggled with `adjoint=True/False` and any additional `**kwargs` are passed on to `torchdiffeq.odeint[_adjoint]`, for example to specify the solver.

### Constructing controls

The other part of this library is a way of constructing paths `X` from data (which may be irregularly sampled with missing values). Linear interpolation and natural cubic splines are supported. For example, for natural cubic splines:
```python
coeffs = natural_cubic_spline_coeffs(t, x)
```
where:
- `t` is a one-dimensional Tensor of shape `(length,)`, giving observation times,
- `X` is a Tensor of shape `(..., length, input_channels)`, where `...` is some number of batch dimensions.

This will compute some coefficients of the natural cubic spline, and handles missing data, and should usually be done as a preprocessing step before training a machine learning model. It returns a tuple of `Tensor`s which you can `torch.save`, `torch.load`, and pass into `torch.util.data.Datasets` and `torch.util.data.DataLoaders` as normal.

During training, this can then be understood by your model:
```
# Inside the model
spline = NaturalCubicSpline(t, coeffs)
```
and then pass `spline` passed to `cdeint` as its `X` argument. See [example.py](./example/example.py).

## Extending the library
If you're interested in extending `torchcontroldiffeq` then have a look at [EXTENDING.md](./EXTENDING.md) for extra help on how to do this.

## Differences to `controldiffeq`
If you've used the previous [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq) library then a couple things have been changed. See [DIFFERENCES.md](./DIFFERENCES.md).
