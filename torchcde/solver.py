import torch
import torchdiffeq


def _check_compatability(X, func, z0, t):
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
    if len(system.shape) < 2:
        raise ValueError("func did not return a tensor with enough dimensions. Expected 2 dimensions, got {} dimensions"
                         " with shape {}. Recall that func must return a (batch of) matrices, describing a linear map."
                         .format(len(system.shape), tuple(system.shape)))
    if system.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(system.shape), tuple(system.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if system.size(-2) != z0.size(-1):
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-2), tuple(z0.shape), z0.size(-1)))
    if system.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as X.derivative "
                         "returned. func returned shape {} (meaning {} channels), whilst X.derivative returned shape "
                         "{} (meaning {} channels)."
                         "".format(tuple(system.shape), system.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))


class _VectorField(torch.nn.Module):
    def __init__(self, X, func):
        """Defines a controlled vector field.

        Arguments:
            X: As cdeint.
            func: As cdeint.
        """
        super(_VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.X = X
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(t, z)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)

        return out


def cdeint(X, func, z0, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(s, z_s) dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        X: The control. This should be a instance of `torch.nn.Module`, with a `derivative` method. For example
            `torchcde.NaturalCubicSpline`. This represents a continuous path derived from the data. The
            derivative at a point will be computed via `X.derivative(t)`, where t is a scalar tensor. The returned
            tensor should have shape (..., input_channels), where '...' is some number of batch dimensions and
            input_channels is the number of channels in the input path.
        func: Should be a callable describing the vector field f(t, z). If using `adjoint=True` (the default), then
            should be an instance of `torch.nn.Module`, to collect the parameters for the adjoint pass. Will be called
            with a scalar tensor t and a tensor z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `X` arguments as above. The '...' corresponds to some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate. Defaults to True.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq (the most common are `rtol`, `atol`,
            `method`, `options`).

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        This will be a tensor of shape (..., len(t), hidden_channels).

    Raises:
        ValueError for malformed inputs.

    Warnings:
        Note that the returned tensor puts the sequence dimension second-to-last, rather than first like in
        `torchdiffeq.odeint`.

        If you need gradients with respect to t (which is quite unusual, but still), then don't use the particular
        combination of:
         - adjoint=True (the default)
         - linear interpolation with reparameterise='none' (the default) to construct X.
        It doesn't work. (For mathematical reasons: the adjoint method requires access to d2X_dt2, which is
        measure-valued for linear interpolation; this doesn't get detected during the solve so you wrongly end up with
        zero gradient.) Switch to either natural cubic splines or reparameterise='bump'.
    """

    # Reduce the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4

    _check_compatability(X, func, z0, t)

    if adjoint:
        try:
            adjoint_params = tuple(kwargs['adjoint_params'])
        except KeyError:
            try:
                adjoint_params = tuple(func.parameters())
            except AttributeError:
                raise ValueError("If using the adjoint, then either `adjoint_params` must be specified, or `func` "
                                 "must be a `torch.nn.Module`, to collect the parameters that gradients are calculated "
                                 "for.")
        try:
            computed_params = tuple(X._torchcde_computed_parameters.values())
        except AttributeError:
            computed_params = ()
        adjoint_params = tuple(param for param in adjoint_params + computed_params if param.requires_grad)
        kwargs['adjoint_params'] = adjoint_params

    vector_field = _VectorField(X=X, func=func)
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    batch_dims = range(1, len(out.shape) - 1)
    out = out.permute(*batch_dims, 0, -1)

    return out
