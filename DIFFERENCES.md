# Differences to `controldiffeq`
We've made a few changes since [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq), either for consistency with other libraries or for technical reasons.

## New features

- Linear interpolation and reparameterised linear interpolation are now available, via `linear_interpolation_coeffs` and `LinearInterpolation`.

- Computing logsignatures over fixed windows is now available, for the log-ODE method, via `logsignature_windows`. (Using this functionality will require installing the [Signatory](https://github.com/patrick-kidger/signatory) package.)

- Can now stack Neural CDEs, so as to drive one Neural CDE with the output of another Neural CDE.

## Interface changes

- `cdeint` now takes `X` rather than `dX_dt` as an argument, which is a little neater.

- The arguments for `cdeint` have been re-ordered. This is for consistency with `torchdiffeq.odeint`.

- The system `func` (the argument to `cdeint`) now also accepts time `t` as an argument when called. (Rather than just the state `z`.)

- You don't need to pass time as an argument to `natural_cubic_spline_coeffs` or `NaturalCubicSpline` any more; an equally spaced parameterisation is picked for you (by default; this can be overridden). Note that if using `NaturalCubicSpline` as an input to a CDE then this change shouldn't really affect things, because of the "reparameterisation invariance" property of CDEs, i.e. that it's just the output values that matter, not the parameterisation.

## Other changes

- The default `rtol` and `atol` for `cdeint` are now reduced down to `1e-4` and `1e-6` respectively. These seem like reasonable choices of CDEs as deafult settings, and still seems to work just fine in terms of empirical performance.

- The sequence dimension of the tensor returned from `cdeint` is now the second-to-last one (rather than the first one) so that the result is now of shape `(...batch dimensions..., length, channels)`. This fixes an inconsistency between the location of the sequence dimension for inputs and outputs. (But as a result is now no longer the same as in `torchdiffeq`.)
