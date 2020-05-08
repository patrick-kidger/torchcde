import torch

from . import path
from . import misc


def _linear_interpolation_coeffs_with_missing_values_scalar(t, X):
    # t and X both have shape (length,)

    not_nan = ~torch.isnan(X)
    path_no_nan = X.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        return torch.zeros(X.size(0), dtype=X.dtype, device=X.device)

    # How to deal with missing values at the start or end of the time series? We impute an observation at the very start
    # equal to the first actual observation made, and impute an observation at the very end equal to the last actual
    # observation made, and then proceed as normal.
    need_new_not_nan = False
    if torch.isnan(X[0]):
        if not need_new_not_nan:
            X = X.clone()
            need_new_not_nan = True
        X[0] = path_no_nan[0]
    if torch.isnan(X[-1]):
        if not need_new_not_nan:
            X = X.clone()
        X[-1] = path_no_nan[-1]

    # TODO


def _linear_interpolation_coeffs_with_missing_values(t, X):
    if len(X.shape) == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _linear_interpolation_coeffs_with_missing_values_scalar(t, X)
    else:
        out_pieces = []
        for p in X.unbind(dim=0):  # TODO: parallelise over this
            out = _linear_interpolation_coeffs_with_missing_values(t, p)
            out_pieces.append(out)
        return misc.cheap_stack(out_pieces, dim=0)


def linear_interpolation_coeffs(t, X):
    """Calculates the knots of the linear interpolation of the batch of controls given.

    Arguments:
        t: One dimensional tensor of times. Must be monotonically increasing.
        X: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        One tensor, which should in turn be passed to `torchcontroldiffeq.LinearInterpolation`.

        See the docstring for `torchcontroldiffeq.natural_cubic_spline_coeffs` for more information on why we do it this
        way.
    """
    path.validate_input(t, X)

    if torch.isnan(X).any():
        return _linear_interpolation_coeffs_with_missing_values(t, X)
    else:
        return X


class LinearInterpolation(path.Path):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, t, coeffs, **kwargs):
        """
        Arguments:
            t: As was passed as an argument to linear_interpolation_coeffs.
            coeffs: As returned by linear_interpolation_coeffs.
        """
        super(LinearInterpolation, self).__init__(**kwargs)

        derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1])

        self._t = path.ComputedParameter(t)
        self._coeffs = path.ComputedParameter(coeffs)
        self._derivs = path.ComputedParameter(derivs)

    def _interpret_t(self, t):
        maxlen = self._coeffs.size(-2) - 1
        index = (t > self._t).sum() - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        prev = self._coeffs[index]
        next = self._coeffs[index + 1]
        return prev + fractional_part * (next - prev)

    def derivative(self, t):
        _, index = self._interpret_t(t)
        return self._derivs[index]
