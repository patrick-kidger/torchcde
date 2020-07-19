import torch
import torchdiffeq


def _normalise_and_check_individual(name, o, t):
    if isinstance(o, torch.nn.Module):
        o = [[t[0], t[-1], o]]
    elif isinstance(o, (tuple, list)):
        for _, _, o_ in o:
            if not isinstance(o_, torch.nn.Module):
                raise ValueError("Each {} must be a torch.nn.Module".format(name))
    else:
        raise ValueError("{} must be a torch.nn.Module or tuple or list".format(name))

    if o[0][0] != t[0]:
        raise ValueError("The first region of integration must start at t[0].")
    if o[-1][1] != t[-1]:
        raise ValueError("The final region of integration must finish at t[-1].")
    prev_t1 = t[0]
    for t0, t1, _ in o:
        if t0 != prev_t1:
            raise ValueError("The regions of integration must be contiguous.")
        prev_t1 = t1
    return o


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


def _normalise_and_check_input(X, func, z0, t):
    X = _normalise_and_check_individual('X', X, t)
    func = _normalise_and_check_individual('func', func, t)
    if len(X) == 1:
        if len(func) != 1:
            # Broadcast X to func
            X = [[t0, t1, X[0][2]] for t0, t1, _ in func]
    else:
        if len(func) == 1:
            # Broadcast func to X
            func = [[t0, t1, func[0][2]] for t0, t1, _ in X]
        else:
            if len(X) == len(func):
                for (t0, t1, _), (t0_, t1_, _) in zip(X, func):
                    if t0 != t0_:
                        raise ValueError("X and func must specify the same regions of integration.")
                    if t1 != t1_:
                        raise ValueError("X and func must specify the same regions of integration.")
            else:
                raise ValueError("X and func must specify the same regions of integration.")

    for (_, _, X_), (_, _, func_) in zip(X, func):
        _check_compatability(X_, func_, z0, t)

    return X, func


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

    def __call__(self, t, z):
        # So what's up with this then?
        #
        # First of all, this only applies if we're using the adjoint method, so this doesn't change anything in the
        # non-adjoint case. In the adjoint case, however, the derivative wrt t is only used to compute the derivative
        # wrt the input times.
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
        # (And if you do want gradients wrt times - don't use linear interpolation!)
        if self.t_not_requires_grad:
            t = t.detach()

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

    X, func = _normalise_and_check_input(X, func, z0, t)

    vector_field = []
    for (t0, t1, X_), (_, _, func_) in zip(X, func):
        vector_field.append([t0, t1, _VectorField(X=X_, func=func_, t_requires_grad=t.requires_grad, adjoint=adjoint)])

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    batch_dims = range(1, len(out.shape) - 1)
    out = out.permute(*batch_dims, 0, -1)

    return out
