import torch
import torchdiffeq


class _VectorField(torch.nn.Module):
    def __init__(self, X, func, t_requires_grad, adjoint):
        """Defines a controlled vector field.

        Arguments:
            X: As cdeint.
            func: As cdeint.
            t_requires_grad: Whether the 't' argument to cdeint requires gradient.
            adjoint: Whether we are using the adjoint method.
        """
        super(_VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.X = X
        self.func = func
        self.t_not_requires_grad = adjoint and not t_requires_grad

    def parameters(self):
        yield from super(_VectorField, self).parameters()
        # Makes sure that the adjoint method sees relevant non-leaf tensors to compute derivatives wrt to.
        for buffer in self.X.buffers():
            if buffer.requires_grad:
                yield buffer

    def __call__(self, t, z):
        # Use tupled input to avoid torchdiffeq doing it for us and breaking the parameters() we've created above.
        z = z[0]

        # So what's up with this then?
        #
        # First of all, this only applies if we're using the adjoint method, so this doesn't change anything in the
        # non-adjoint case. In the adjoint case, however, the derivative wrt t is only used to compute the derivative
        # wrt the input times, and the derivative wrt z is only used to compute the derivative wrt the initial z0.
        #
        # By default torchdiffeq computes all of these gradients regardless, and any ones that aren't needed just get
        # discarded. So for one thing, detaching here gives us a speedup.
        #
        # More importantly, however: the fact that it's computing these gradients affects adaptive step size solvers, as
        # the solver tries to resolve the gradients wrt these additional arguments. In the particular case of linear
        # interpolation, this poses a problem, as the derivative wrt t doesn't exist. (Or rather, it's measure-valued,
        # which is the same thing as far as things are concerned here.) This breaks the adjoint method.
        #
        # As it's generally quite rare to compute derivatives wrt the times, this is the fix: most of the time we just
        # tell torchdiffeq that we don't have a gradient wrt that input, so it doesn't bother calculating it in the
        # first place.
        #
        # (And if you do want gradients wrt times - just don't use linear interpolation!)
        t_with_grad = t
        if self.t_not_requires_grad:
            t = t.detach()

        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(t, z)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)

        # Workaround for PyTorch bug #39784
        dummy = torch.as_strided(t_with_grad, (), ())
        dummy = dummy + torch.as_strided(z, (), ())
        for param in self.parameters():
            dummy = dummy + torch.as_strided(param, (), ())
        dummy = dummy * 0
        out = out + dummy

        return (out,)


def cdeint(X, func, z0, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(s, z_s) dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        X: The control. This should be a instance of `torch.nn.Module`, with a `derivative` method. For example
            `torchcontroldiffeq.NaturalCubicSpline`. This represents a continuous path derived from the data. The
            derivative at a point will be computed via `X.derivative(t)`, where t is a scalar tensor. The returned
            tensor should have shape (..., input_channels), where '...' is some number of batch dimensions and
            input_channels is the number of channels in the input path.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(t, z). Will be called with a
            scalar tensor t and a tensor z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `X` arguments as above. The '...' corresponds to some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate. Defaults to True.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        This will be a tensor of shape (..., len(t), hidden_channels).

    Raises:
        ValueError for malformed inputs.

    Warnings:
        Note that the returned tensor puts the sequence dimension second-to-last, rather than first like in
        `torchdiffeq.cdeint`.

        If you need gradients with respect to t (which is quite unusual, but still), then don't use the particular
        combination of:
         - adjoint=True (the default)
         - linear interpolation, with reparameterise=False, to construct X.
        It doesn't work. (For mathematical reasons: the adjoint method requires access to d2X_dt2, which is
        measure-valued for linear interpolation; this doesn't get detected during the solve so you wrongly end up with
        zero gradient.) Switch to either natural cubic splines or reparameterise=True.
    """

    # Change the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-3
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-3

    if not isinstance(X, torch.nn.Module):
        raise ValueError("X must be an instance of torch.nn.Module, so that we can correctly find the parameters for "
                         "the adjoint method.")
    if not hasattr(X, 'derivative'):
        raise ValueError("X must have a 'derivative' method.")
    control_gradient = X.derivative(t[0].detach())
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("X.derivative did not return a tensor with the same number of batch dimensions as z0. "
                         "X.derivative returned shape {} (meaning {} batch dimensions)), whilst z0 has shape {} "
                         "(meaning {} batch dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    system = func(t[0], z0)
    if system.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(system.shape), tuple(system.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if system.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if system.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as X.derivative "
                         "returned. func returned shape {} (meaning {} channels), whilst X.derivative returned shape "
                         "{} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))

    vector_field = _VectorField(X=X, func=func, t_requires_grad=t.requires_grad, adjoint=adjoint)
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # Note how we pass in and unwrap a tuple to avoid torchdiffeq wrapping vector_field in something that isn't
    # computed-parameter aware.
    # I don't like depending on an implementation detail of torchdiffeq like this but I don't see many other options
    # that don't involve directly modifying torch.nn.Module (probably best to stay away from that).
    out = odeint(func=vector_field, y0=(z0,), t=t, **kwargs)[0]

    batch_dims = range(1, len(out.shape) - 1)
    out = out.permute(*batch_dims, 0, -1)

    return out
