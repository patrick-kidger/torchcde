# Differences to `controldiffeq`
We've made a couple changes since [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq), either for consistency with other libraries or for technical reasons.

- `cdeint` now takes `X` rather than `dX_dt` as an argument. Besides being a little easier to use, this change is actually what makes it possible on a technical level to stack Neural CDEs on top of one another (that is, driving one Neural CDE with the output of another Neural CDE).

- The arguments for `cdeint` have been re-ordered. This is for consistency with `torchdiffeq.odeint`.

- The sequence dimension of the tensor returned from `cdeint` is now the second-to-last one (rather than the first one) so that the result is now of shape `(...batch dimensions..., length, channels)`. This fixes an inconsistency between the location of the sequence dimension for inputs and outputs.