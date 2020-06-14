import math
import torch

from . import path
from . import misc


_two_pi = 2 * math.pi
_inv_two_pi = 1 / _two_pi


def _linear_interpolation_coeffs_with_missing_values_scalar(t, x):
    # t and X both have shape (length,)

    not_nan = ~torch.isnan(x)
    path_no_nan = x.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        return torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

    if path_no_nan.size(0) == x.size(0):
        # Every entry is not-NaN, so just return.
        return x

    x = x.clone()
    # How to deal with missing values at the start or end of the time series? We impute an observation at the very start
    # equal to the first actual observation made, and impute an observation at the very end equal to the last actual
    # observation made, and then proceed as normal.
    if torch.isnan(x[0]):
        x[0] = path_no_nan[0]
    if torch.isnan(x[-1]):
        x[-1] = path_no_nan[-1]

    nan_indices = torch.arange(x.size(0), device=x.device).masked_select(torch.isnan(x))

    if nan_indices.size(0) == 0:
        # We only had missing values at the start or end
        return x

    prev_nan_index = nan_indices[0]
    prev_not_nan_index = prev_nan_index - 1
    prev_not_nan_indices = [prev_not_nan_index]
    for nan_index in nan_indices[1:]:
        if prev_nan_index != nan_index - 1:
            prev_not_nan_index = nan_index - 1
        prev_nan_index = nan_index
        prev_not_nan_indices.append(prev_not_nan_index)

    next_nan_index = nan_indices[-1]
    next_not_nan_index = next_nan_index + 1
    next_not_nan_indices = [next_not_nan_index]
    for nan_index in reversed(nan_indices[:-1]):
        if next_nan_index != nan_index + 1:
            next_not_nan_index = nan_index + 1
        next_nan_index = nan_index
        next_not_nan_indices.append(next_not_nan_index)
    next_not_nan_indices = reversed(next_not_nan_indices)
    for prev_not_nan_index, nan_index, next_not_nan_index in zip(prev_not_nan_indices,
                                                                 nan_indices,
                                                                 next_not_nan_indices):
        prev_stream = x[prev_not_nan_index]
        next_stream = x[next_not_nan_index]
        prev_time = t[prev_not_nan_index]
        next_time = t[next_not_nan_index]
        time = t[nan_index]
        ratio = (time - prev_time) / (next_time - prev_time)
        x[nan_index] = prev_stream + ratio * (next_stream - prev_stream)

    return x


def _linear_interpolation_coeffs_with_missing_values(t, x):
    if x.ndimension() == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _linear_interpolation_coeffs_with_missing_values_scalar(t, x)
    else:
        out_pieces = []
        for p in x.unbind(dim=0):  # TODO: parallelise over this
            out = _linear_interpolation_coeffs_with_missing_values(t, p)
            out_pieces.append(out)
        return misc.cheap_stack(out_pieces, dim=0)


def linear_interpolation_coeffs(t, x):
    """Calculates the knots of the linear interpolation of the batch of controls given.

    Arguments:
        t: One dimensional tensor of times. Must be monotonically increasing.
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tensor, which should in turn be passed to `torchcontroldiffeq.LinearInterpolation`.

        See the docstring for `torchcontroldiffeq.natural_cubic_spline_coeffs` for more information on why we do it this
        way.
    """
    path.validate_input(t, x)

    if torch.isnan(x).any():
        x = _linear_interpolation_coeffs_with_missing_values(t, x.transpose(-1, -2)).transpose(-1, -2)
    return x


class LinearInterpolation(path.Path):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, t, coeffs, reparameterise=True, **kwargs):
        """
        Arguments:
            t: As passed to linear_interpolation_coeffs.
            coeffs: As returned by linear_interpolation_coeffs.
            reparameterise: If True (the default), then the speed of the interpolation will increase between knots and
                slow down near them. This helps adaptive step solvers resolve the path more easily.
        """
        super(LinearInterpolation, self).__init__(**kwargs)

        derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1]).unsqueeze(-1)

        self._t = path.ComputedParameter(t)
        self._coeffs = path.ComputedParameter(coeffs)
        self._derivs = path.ComputedParameter(derivs)
        # assert 0 <= reparametrise, "reparameterise must be between 0 and 1"
        # assert reparametrise <= 1, "reparameterise must be between 0 and 1"
        self._reparameterise = reparameterise

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._derivs.dtype)
        maxlen = self._derivs.size(-2) - 1
        # TODO: switch to a log search not a linear search
        index = (t.unsqueeze(-1) > self._t).sum(dim=-1) - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        prev_coeff = self._coeffs[..., index, :]
        next_coeff = self._coeffs[..., index + 1, :]
        prev_t = self._t[index]
        next_t = self._t[index + 1]
        diff_t = next_t - prev_t
        if self._reparameterise:
            fractional_part = fractional_part - diff_t * _inv_two_pi * torch.sin(_two_pi * fractional_part / diff_t)
        return prev_coeff + fractional_part * (next_coeff - prev_coeff) / diff_t.unsqueeze(-1)

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        deriv = self._derivs[..., index, :]

        if self._reparameterise:
            prev_t = self._t[index]
            next_t = self._t[index + 1]
            diff_t = next_t - prev_t

            fractional_part = fractional_part / diff_t
            mult = 1 - torch.cos(_two_pi * fractional_part)

            deriv = deriv * mult
        return deriv
