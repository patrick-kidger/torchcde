# Extending `torchcde`
This file contains some 'developer notes' which you might find helpful if you want to extend the library for your own purposes.

## Defining your own ways of creating paths (i.e. the `X` argument of `cdeint`.)

The `X` argument to `torchcde.cdeint` will typically be one of the default interpolation strategies available in this library (linear or natural cubic spline). For most purposes they are probably sufficient.

But there's no reason you can't define your own!

#### Things to do:

- `X` should be an instance of `torch.nn.Module`, and you should implement a `derivative(t)` method, taking a scalar tensor `t`.

- `X` probably depends on some tensors that you've computed from the data. Let the adjoint method know that you want gradients with respect to these by registering them via `torchcde.register_computed_parameter(...)`, in `__init__(...)`. (See `interpolation_linear.py` for an example of how it's done.)
