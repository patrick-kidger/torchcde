# Differences to `controldiffeq`
We've made a few changes since [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq), either for consistency with other libraries or for technical reasons.

## New features

- Linear interpolation and reparameterised linear interpolation is now available. (`linear_interpolation_coeffs` and `LinearInterpolation`).

- Computing logsignatures over fixed windows is now available. (`logsignature_windows`) This will require installing the [Signatory](https://github.com/patrick-kidger/signatory) package to use this functionality though.

- Can now stack Neural CDEs, so as to drive out Neural CDE with the output of another Neural CDE.

## Interface changes

- `cdeint` now takes `X` rather than `dX_dt` as an argument. Besides being a little easier to use, this change actually makes it possible on a technical level to stack Neural CDEs on top of one another.

- The arguments for `cdeint` have been re-ordered. This is for consistency with `torchdiffeq.odeint`.

- The system `func` (the argument to `cdeint`) now also accepts time `t` as an argument when called. (Rather than just the state `z`.)

- The sequence dimension of the tensor returned from `cdeint` is now the second-to-last one (rather than the first one) so that the result is now of shape `(...batch dimensions..., length, channels)`. This fixes an inconsistency between the location of the sequence dimension for inputs and outputs.
