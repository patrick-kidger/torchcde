<h1 align='center'>torchcde</h1>
<h2 align='center'>Differentiable GPU-capable solvers for CDEs</h2>

This library provides differentiable GPU-capable solvers for controlled differential equations (CDEs). Backpropagation through the solver or via the adjoint method is supported; the latter allows for improved memory efficiency.

In particular this allows for building [Neural Controlled Differential Equation](https://github.com/patrick-kidger/NeuralCDE) models, which are state-of-the-art models for (irregular) time series; they can be thought of as a "continuous time RNN".

_Powered by the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library._

---

<p align="center">
<img align="middle" src="./imgs/main.png" width="666" />
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
cdeint(X, func, z0, t, adjoint, **kwargs)
```
where letting `...` denote an arbitrary number of batch dimensions:
* `X` is a `torch.nn.Module` with method `derivative`, such that `X.derivative(t)` is a Tensor of shape `(..., input_channels)`,
* `func` is a `torch.nn.Module`, such that `func(t, z)` returns a Tensor of shape `(..., hidden_channels, input_channels)`,
* `z0` a Tensor of shape `(..., hidden_channels)`,
* `t` is a one-dimensional Tensor of times to output `z` at.
* `adjoint` is a boolean (defaulting to `True`).

Adjoint backpropagation (which is slower but more memory efficient) can be toggled with `adjoint=True/False` and any additional `**kwargs` are passed on to `torchdiffeq.odeint[_adjoint]`, for example to specify the solver.

### Constructing controls

 A very common scenario is to construct the continuous control`X` from discrete data (which may be irregularly sampled with missing values). To support this, we provide three interpolation schemes:
* Linear interpolation
* Reparameterised linear interpolation
* Natural cubic splines

_Note that if for some reason you already have a continuous control `X` then you won't need an interpolation scheme at all!_

Natural cubic splines were used in the original [Neural CDE paper](https://arxiv.org/abs/2005.08926). We now recommend linear interpolation, whilst telling the solver about the grid points, as usually a slightly better choice.

To do this:
```python
coeff = linear_interpolation_coeffs(t, x)

# coeff is a torch.Tensor you can save, load,
# pass through Datasets and DataLoaders etc.

X = LinearInterpolation(t, coeff)
```
where:
* `t` is a one-dimensional Tensor of shape `(length,)`, giving observation times,
* `x` is a Tensor of shape `(..., length, input_channels)`, where `...` is some number of batch dimensions. Missing data should be represented as a `NaN`.

Usually the first line should be done as a preprocessing step, whilst the second line should be inside the forward pass of your model. See [example.py](./example/example.py) for a worked example.

Then call:
```python
cdeint(X=X, ...
       method='dopri5',
       options=dict(grid_points=X.grid_points, eps=1e-5))
```
Linear interpolation produces sharp changes at each interpolation point. Setting `grid_points` and `eps` like this tells the solver where the are, so that it can adapt to them.

_See the [further documentation](#further-documentation) at the bottom for more discussion on what's being done here, and for discussion on the other interpolation schemes._

## Differences to `controldiffeq`
If you've used the previous [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq) library then a couple things have been changed. See [DIFFERENCES.md](./DIFFERENCES.md).

## Extending the library
If you're interested in extending `torchcde` then have a look at [EXTENDING.md](./EXTENDING.md) for extra help on how to do this.

## Further documentation
Here we discuss:
* The importance of the `grid_points` and `eps` options for adaptive solvers.
* The use of fixed solvers.
* Other interpolation methods, and the differences between them.

And some more advanced functionality:
* Stacking CDEs (i.e. controlling one by the output of another).
* Computing logsignatures for the log-ODE method.

#### `grid_points` and `eps` with adaptive solvers
If using linear interpolation, then integrating the CDE naively can be difficult: we get a jump in the derivative at each interpolation point, and this slows adaptive step size solvers down. First they have to slow down to resolve the point - and then they have to figure out that they can speed back up again afterwards.

We can help them out by telling them about the prescence of these jumps, so that they don't have to discover it for themselves.

We do this by passing the `grid_points` option with either the `dopri5` or `dopri8` solvers, which specify the points at which these jumps exist, so that the solver can place its integration points directly on the jump. The `torchde.LinearInterpolation` class provides a helper `.grid_points` property that can be passed to set the the grid points correctly, as in the examples.

There's one more important thing to include: the `eps` argument. Recall that we're solving the differential equation
```
dz/dt(t) = f(t, z)dX/dt(t)     z(t_0) = z0
```
where `X` is piecewise linear. Thus `dX/dt` is piecewise constant; it has jumps. Thus we don't want to place our integration points exactly on the jump, as `X` isn't consistently defined there. We need to place integration points just to the left, and just to the right, of that jump. `eps` specifices how much to shift to the left or right, so it has just has to be some very small number above zero.

#### Fixed solvers
Solving CDEs (regardless of the choice of interpolation scheme in a Neural CDE) with fixed solvers like `euler`, `midpoint`, `rk4` etc. is pretty much exactly the same as solving an ODE with a fixed solver. Just make sure to set the `step_size` option to something sensible; for example the smallest gap between times:
```python
X = LinearInterpolation(t, coeffs)
cdeint(
    X=X, t=t[[0, -1]], func=..., method='rk4',
    options=dict(step_size=(t[1:] - t[:-1]).min())
)
``` 

#### Different interpolation methods
* Linear interpolation: these are causal, but are not smooth, which makes them hard to integrate - unless we tell the solver about the difficult points, in which case they become very easy to integrate!
```python
coeff = linear_interpolation_coeffs(t, x)
X = LinearInterpolation(t, coeff)
cdeint(X=X, ...,
       method='dopri5',
       options=dict(grid_points=X.grid_points, eps=1e-5))
```

* Reparameterised linear interpolation: these are causal, and quite smooth, making them reasonably easy to integrate.
```python
coeff = linear_interpolation_coeffs(t, x)
X = LinearInterpolation(t, coeff, reparameterise='bump')
cdeint(X=X, ...)  # no options necessary
```

* Natural cubic splines: these were a simple choice used in the original Neural CDE paper. They are non-causal, but are quite smooth, which makes them easy to integrate.
```python
coeffs = natural_cubic_splines_coeffs(t, x)
X = NaturalCubicSpline(t, coeffs)
cdeint(X=X, ...)  # no options necessary
```

See [interpolation_comparison.py](./example/interpolation_comparison.py) for a comparison of the speed of each of these with adapative step size solvers.

_To the best of our knowledge there is nearly no reason to use anything other than grid-aware linear interpolation. It is causal, it is faster than any other method due to its ease of integration (whether using fixed or adaptive solvers), and the accuracy of Neural CDE models don't seem to be affected. In the very low accuracy regime (`atol,rtol=1e-2` or so) then cubic splines may be ever-so-slightly faster, but that's it._

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

_For example, this could be used to create multi-layer Neural CDEs, just like multi-layer RNNs. Although as of writing this, no-one seems to have tried this yet!_

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
