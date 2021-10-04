try:
    import signatory
except ImportError:
    class DummyModule:
        def __getattr__(self, item):
            raise ImportError("signatory has not been installed. Please install it from "
                              "https://github.com/patrick-kidger/signatory to use the log-ODE method.")
    signatory = DummyModule()
import torch

from . import interpolation_linear
from . import misc


def _logsignature_windows(x, depth, window_length, t, _version):
    t = misc.validate_input_path(x, t)

    # slightly roundabout way of doing things (rather than using arange) so that it's constructed differentiably
    timespan = t[-1] - t[0]
    num_pieces = (timespan / window_length).ceil().to(int).item()
    end_t = t[0] + num_pieces * window_length
    new_t = torch.linspace(t[0], end_t, num_pieces + 1, dtype=t.dtype, device=t.device)
    new_t = torch.min(new_t, t.max())

    t_index = 0
    new_t_unique = []
    new_t_indices = []
    for new_t_elem in new_t:
        while True:
            lequal = (new_t_elem <= t[t_index])
            close = new_t_elem.allclose(t[t_index])
            if lequal or close:
                break
            t_index += 1
        new_t_indices.append(t_index + len(new_t_unique))
        if close:
            continue
        new_t_unique.append(new_t_elem.unsqueeze(0))

    batch_dimensions = x.shape[:-2]

    missing_X = torch.full((1,), float('nan'), dtype=x.dtype, device=x.device).expand(*batch_dimensions, 1, x.size(-1))
    if len(new_t_unique) > 0:  # no-op if len == 0, so skip for efficiency
        t, indices = torch.cat([t, *new_t_unique]).sort()
        x = torch.cat([x, missing_X], dim=-2)[..., indices.clamp(0, x.size(-2)), :]

    # Fill in any missing data linearly (linearly because that's what signatures do in between observations anyway)
    # and conveniently that's what this already does. Here 'missing data' includes the NaNs we've just added.
    x = interpolation_linear.linear_interpolation_coeffs(x, t)

    # Flatten batch dimensions for compatibility with Signatory
    flatten_X = x.reshape(-1, x.size(-2), x.size(-1))
    first_increment = torch.zeros(*batch_dimensions, signatory.logsignature_channels(x.size(-1), depth), dtype=x.dtype,
                                  device=x.device)
    first_increment[..., :x.size(-1)] = x[..., 0, :]
    logsignatures = [first_increment]
    compute_logsignature = signatory.Logsignature(depth=depth)
    for index, next_index, time, next_time in zip(new_t_indices[:-1], new_t_indices[1:], new_t[:-1], new_t[1:]):
        logsignature = compute_logsignature(flatten_X[..., index:next_index + 1, :])
        logsignature = logsignature.view(*batch_dimensions, -1)
        if _version == 0:
            logsignature = logsignature * (next_time - time)
        elif _version == 1:
            pass
        else:
            raise RuntimeError
        logsignatures.append(logsignature)

    logsignatures = torch.stack(logsignatures, dim=-2)
    logsignatures = logsignatures.cumsum(dim=-2)

    if _version == 0:
        return logsignatures, new_t
    elif _version == 1:
        return logsignatures
    else:
        raise RuntimeError


def logsignature_windows(x, depth, window_length, t=None):
    """Calculates logsignatures over multiple windows, for the batch of controls given, as in the log-ODE method.

    ********************
    DEPRECATED: this now exists for backward compatibility. For new projects please use `logsig_windows` instead,
    which has a corrected rescaling coefficient.
    ********************

    This corresponds to a transform of the time series, and should be used prior to applying one of the interpolation
    schemes.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        depth: What depth to compute the logsignatures to.
        window_length: How long a time interval to compute logsignatures over.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]).

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tuple of two tensors, which are the values and times of the transformed path.
    """
    return _logsignature_windows(x, depth, window_length, t, _version=0)


def logsig_windows(x, depth, window_length, t=None):
    """Calculates logsignatures over multiple windows, for the batch of controls given, as in the log-ODE method.

    This corresponds to a transform of the time series, and should be used prior to applying one of the interpolation
    schemes.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an `input_channels`-dimensional real vector space,
            with `length`-many observations. Missing values are supported, and should be represented as NaNs.
        depth: What depth to compute the logsignatures to.
        window_length: How long a time interval to compute logsignatures over.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            `tensor([0., 1., ..., length - 1])`.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tensor, which are the values of the transformed path. Times are _not_ returned: the return value is
        always scaled such that the corresponding times are just `tensor([0., 1., ..., length - 1])`.
    """
    return _logsignature_windows(x, depth, window_length, t, _version=1)
