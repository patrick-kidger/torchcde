try:
    import signatory
except ImportError:
    class DummyModule:
        def __getattr__(self, item):
            raise ImportError("signatory has not been installed. Please install it from "
                              "https://github.com/patrick-kidger/signatory to use the log-ODE method.")
    signatory = DummyModule()
import torch

from . import linear_interpolation
from . import path


def log_ode_coeffs(t, X, depth, window_length):
    """Calculates logsignatures for use in the log-ODE method, for the batch of controls given.

    Arguments:
        t: One dimensional tensor of times. Must be monotonically increasing.
        X: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        depth: What depth to compute the logsignatures to.
        window_length: How long a time interval to compute logsignatures over.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tuple of two tensors, which should in turn be passed to `torchcontroldiffeq.LogODE`.

        See the docstring for `torchcontroldiffeq.natural_cubic_spline_coeffs` for more information on why we do it this
        way.
    """
    path.validate_input(t, X)

    if isinstance(window_length, torch.Tensor):
        window_length = window_length.item()
    new_t = torch.arange(t[0].item(), t[-1].item() + window_length, window_length, dtype=t.dtype, device=t.device)
    new_t[-1].clamp_(t[0].item(), t[-1].item())

    t_index = 0
    new_t_list = []
    new_t_indices = []
    for new_t_elem in new_t:
        while new_t_elem > t[t_index]:
            t_index += 1
        new_t_indices.append(t_index + len(new_t_list))
        if new_t_elem.allclose(t[t_index]):
            continue
        if new_t_elem.allclose(t[t_index - 1]):
            continue
        new_t_list.append(new_t_elem)
    new_t_unique = torch.cat(new_t_list)

    batch_dimensions = X.shape[:-2]
    missing_X = torch.full((*batch_dimensions, 1, X.size(-1)), float('nan'), dtype=X.dtype, device=X.device)

    t, indices = torch.cat([t, new_t_unique]).sort()
    X = torch.cat([X, missing_X], dim=-2)[indices.clamp(0, len(t) - 1)]

    # Fill in any missing data linearly (linearly because that's what signatures do in between observations anyway)
    # and conveniently that's what this already does
    X = linear_interpolation.linear_interpolation_coeffs(t, X)

    # Flatten batch dimensions for compatibility with Signatory
    X = X.view(-1, X.size(-2), X.size(-1))

    logsignatures = []
    Logsignature = signatory.Logsignature(depth=depth)
    for index, next_index in zip(new_t_indices[:-1], new_t_indices[1:]):
        logsignature = Logsignature(X[..., index:next_index + 1, :])
        logsignatures.append(logsignature.view(*batch_dimensions, -1))

    logsignatures = torch.stack(logsignatures, dim=-2)

    return new_t, logsignatures


class LogODE(path.Path):
    """Describes the logsignatures of some data as in the log-ODE method."""

    def __init__(self, coeffs, **kwargs):
        """
        Arguments:
            coeffs: As returned by log_ode_coeffs.
        """
        super(LogODE, self).__init__(**kwargs)

        t, logsignatures = coeffs

        self._t = path.ComputedParameter(t)
        self._logsignatures = path.ComputedParameter(logsignatures)

    def _interpret_t(self, t):
        maxlen = self._logsignatures.size(-2) - 1
        index = (t > self._t).sum() - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        # just computing an integral here basically.
        fractional_part, index = self._interpret_t(t)
        t_diff = self._t[1:index + 1] - self._t[:index]
        scaled_logsignatures = t_diff.unsqueeze(-1) * self._logsignatures[..., :index, :]
        return scaled_logsignatures.sum(dim=-2) + fractional_part * self._logsignatures[..., index, :]

    def derivative(self, t):
        _, index = self._interpret_t(t)
        return self._logsignatures[..., index, :]
