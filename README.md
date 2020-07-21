<h1 align='center'>torchcde</h1>
<h2 align='center'>Differentiable GPU-capable solvers for CDEs</h2>

This library provides differentiable GPU-capable solvers for controlled differential equations (CDEs). Backpropagation through the solver or via the adjoint method is supported; the latter allows for improved memory efficiency.

In particular this allows for building [Neural Controlled Differential Equation](https://github.com/patrick-kidger/NeuralCDE) models, which are state-of-the-art models for (irregular) time series; they can be thought of as a "continuous time RNN".

_Powered by the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library._

---

<p align="center">
<img align="middle" src="./imgs/main.png" width="300" />
</p>

## Installation

```bash
pip install git+https://github.com/patrick-kidger/torchcde.git
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
dz(t) = f(t, z(t))dX(t)     z(t_0) = z0
```

The goal is to find the response `z` driven by the control `X`. This can be re-written as the following differential equation:
```
dz/dt(t) = f(t, z)dX/dt(t)     z(t_0) = z0
```
where the right hand side describes a matrix-vector product between `f(t, z)` and `dX/dt(t)`.

This is solved by
```python
cdeint(X, func, z0, t, adjoint=True, **kwargs)
```
where letting `...` denote an arbitrary number of batch dimensions:
* `X` is a `torch.nn.Module` with method `derivative`, such that `X.derivative(t)` is a Tensor of shape `(..., input_channels)`,
* `func` is a `torch.nn.Module`, such that `func(t, z)` returns a Tensor of shape `(..., hidden_channels, input_channels)`,
* `z0` a Tensor of shape `(..., hidden_channels)`,
* `t` is a one-dimensional Tensor of times.

Adjoint backpropagation (which is slower but more memory efficient) can be toggled with `adjoint=True/False` and any additional `**kwargs` are passed on to `torchdiffeq.odeint[_adjoint]`, for example to specify the solver.

### Constructing controls

 A very common scenario is to construct the continuous control`X` from discrete data (which may be irregularly sampled with missing values). To support this, we provide several interpolation schemes:
* Linear interpolation
* Reparameterised linear interpolation
* Multiple-region linear interpolation
* Natural cubic splines

Natural cubic splines were used in the original [Neural CDE paper](https://arxiv.org/abs/2005.08926). We now recommend multiple-region linear interpolation as the usual best choice.

_Note that if for some reason you already have a continuous control `X` then you won't need an interpolation scheme at all!_

To use multiple-region linear interpolation:
```python
coeff = linear_interpolation_coeffs(t, x)

# coeff is a torch.Tensor you can save, load,
# pass through Datasets and DataLoaders etc.

interp = LinearInterpolation(t, coeff)
```
where:
* `t` is a one-dimensional Tensor of shape `(length,)`, giving observation times,
* `x` is a Tensor of shape `(..., length, input_channels)`, where `...` is some number of batch dimensions. Missing data should be represented as a `NaN`.

Usually the first line should be done as a preprocessing step, whilst the second line should be inside the forward pass of your model. See [example.py](./example/example.py) for a worked example.

Then call `cdeint(X=interp, ...)` for just linear interpolation, or (typically much faster), `cdeint(X=interp.multiple_region(), ...)` to use multiple-region linear interpolation.

_See the [further documentation](##-further-documentation) at the bottom for a discussion of the other interpolation schemes._

## Extending the library
If you're interested in extending `torchcde` then have a look at [EXTENDING.md](./EXTENDING.md) for extra help on how to do this.

## Differences to `controldiffeq`
If you've used the previous [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq) library then a couple things have been changed. See [DIFFERENCES.md](./DIFFERENCES.md).

## Further documentation
`torchcde` also provides a few pieces of more advanced functionality. Here we discuss:
* Other interpolation methods, and the differences between them.
* Using piecewise `func`.
* Stacking CDEs (i.e. controlling one by the output of another).
* Computing logsignatures for the log-ODE method.

#### Different interpolation methods
* Linear interpolation: these are causal, but are not smooth, which makes them hard to integrate.
```python
coeff = linear_interpolation_coeffs(t, x)
interp = LinearInterpolation(t, coeff)
```

* Reparameterised linear interpolation: these are causal, and quite smooth, making them reasonably easy to integrate.
```python
coeff = linear_interpolation_coeffs(t, x)
interp = LinearInterpolation(t, coeff, reparameterise=True)
```

* Multiple-region linear interpolation: this is just linear interpolation, but we help out of the numerical solver by telling it where the sharp corners are.
```python
coeff = linear_interpolation_coeffs(t, x)
interp = LinearInterpolation(t, coeff).multiple_region()
# interp is now a list, so call multiple_region() just before
# passing it to cdeint; not before.
```

* Natural cubic splines: these were a simple choice used in the original Neural CDE paper. They are non-causal, but are quite smooth, which makes them easy to integrate.
```python
coeffs = natural_cubic_splines_coeffs(t, x)
interp = NaturalCubicSpline(t, coeffs)
```

See [interpolation_comparison.py](./example/interpolation_comparison.py) for a comparison of the speed of each of these with adapative step size solvers.

_To the best of our knowledge there is essentially no reason to use anything other than multiple-region linear interpolation. It is causal, it is faster than any other method due to its ease of integration (whether using fixed or adaptive solvers), and the accuracy of Neural CDE models don't seem to be affected._

#### Using piecewise `func`
It may be that `func` has some piecewise structure, where it has different behaviour at different times `t`. This can be handled just by adding an `if` statement inside its `forward` pass, but then the numerical solver has to discover these discontinuities for itself, which may slow things down.

Instead, you can explicitly tell the solver about the piecewise structure like this:
```python
piece_func1 = ...  # some torch.nn.Module
piece_func2 = ...  # some torch.nn.Module
piece_func3 = ...  # some torch.nn.Module
t = torch.tensor([0., 1.])
func = ((0., 0.2, piece_func1),
        (0.2, 0.9, piece_func2),
        (0.9, 1.), piece_func3))
torchcde.cdeint(func=func, t=t, ...)
```
where we tell the solver to use `piece_func1` on the time interval `0, 0.2`, to use `piece_func2` on the time interval `0.2, 0.9`, and `piece_func3` on the time interval `0.9, 1`.

We refer to each of these time intervals as a region of integration.

#### Stacking CDEs
You may wish to use the output of one CDE to control another. That is, to solve the coupled CDEs:
```
du(t) = g(t, u(t)dz(t)      u(t_0) = u0
dz(t) = f(t, z(t))dX(t)     z(t_0) = z0
```

There are two ways to do this. The first way is to put everything inside a single `cdeint` call, by solving the system
```
v = [u, z]
v0 = [u0, z0]
h(t, v) = [g(t, u)f(t, z), f(t, z)]

dv(t) = h(t, v(t))dX(t)      v(t_0) = v0
```
and using `cdeint` as normal. This is probably simpler, but forces you to use the same solver for the whole system.

The second way is to have `cdeint` output `z(t)` at multiple times `t`, interpolate the discrete output into a continuous path, and then call `cdeint` again. This is probably less memory efficient, but allows for different choices of solver for each call to `cdeint`.

_For example, this could be used to create multi-layer Neural CDEs, just like multi-layer RNNs. Although at the time of writing, no-one has tried this yet!_

#### The log-ODE method
This is a way of reducing the length of data by using extra channels. (For example, this may help train Neural CDE models faster, as the extra channels can be parallelised, but extra length cannot.)

This is done by splitting the control `X` up into windows, and computing the _logsignature_ of the control over each window. The logsignature is a transform known to extract the information that is most important to describing how `X` controls a CDE.

This is supported by the `logsignature_windows` function, which takes in data, and produces a transformed path in logsignature space:
```python
batch, length, channels = 1, 100, 2
t = torch.linspace(0, 1, length)
x = torch.rand(batch, length, channels)
depth, window = 3, 0.2
t, x = logsignature_windows(t, x, depth, window)
# use t and x as you would normally: interpolate, etc.
```

See the docstring of `logsignature_windows` for more information.

_Note that this requires installing the [Signatory](https://github.com/patrick-kidger/signatory) package._
