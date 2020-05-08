# Defining your own ways of creating paths (i.e. the `X` argument of `cdeint`.)

The `X` argument to `torchcontroldiffeq.cdeint` will typically be one of the default interpolation strategies available in this library (linear, natural cubic spline, log-ODE). For most purposes they are probably sufficient.

But there's no reason you can't define your own! There are two technical caveats to observe to make sure things work, however. (The existing linear interpolation may make a good example; see the `linear_interpolation.py` file.)

## The rules

- First of all, whatever your interpolation / curve-fitting / etc. strategy is, it should inherit from `torchcontroldiffeq.Path`, and implement the `derivative(t)` (and optionally `evaluate(t)`) method.

- Second, your path probably depends on some tensors that you've computed. When recording these tensors in `__init__`, make sure to wrap them in `torchcontroldiffeq.ComputedParameter`s. (See how it's done in the existing strategies.)

## Why these rules?

These rules are necessary to make sure that the adjoint-based backpropagation works correctly. Wrapping any computed tensors in `ComputedParameter`, and assigning them to something subclassing `Path`, will between them ensure that these tensors get registered with the autograd framework in the correct way.