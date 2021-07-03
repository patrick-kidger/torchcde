import torch
from torchcde.interpolation_linear import linear_interpolation_coeffs


def _setup_hermite_cubic_coeffs_w_backward_differences(times, coeffs, derivs, device=None):
    """Compute backward hermite from linear coeffs."""
    x_prev = coeffs[..., :-1, :]
    x_next = coeffs[..., 1:, :]
    # Let x_0 - x_{-1} = x_1 - x_0
    derivs_prev = torch.cat((derivs[..., [0], :], derivs[..., :-1, :]), axis=-2)
    derivs_next = derivs
    x_diff = x_next - x_prev
    t_diff = (times[1:] - times[:-1]).unsqueeze(-1)
    # Coeffs
    a = x_prev
    b = derivs_prev
    c = 3 * (x_diff / t_diff ** 2 - b / t_diff) - (derivs_next - derivs_prev) / t_diff
    d = (1 / (3 * t_diff ** 2)) * (derivs_next - b) - (2 * c) / (3 * t_diff)
    coeffs = torch.cat([a, b, 2 * c, 3 * d], dim=-1).to(device)
    return coeffs


def hermite_cubic_coefficients_with_backward_differences(x, t=None):
    """Computes the coefficients for hermite cubic splines with backward differences.

    Arguments:
        See `linear_interpolation_coeffs` from `torchcde.interpolation.interpolation_linear`.

    Returns:
        A tensor, which should in turn be passed to `torchcde.HermiteCubicSplinesWithBackwardDifferences`.

        See the docstring for `torchcde.natural_cubic_coeffs` for more information on why we do it this way.
    """
    # Lineaer coeffs
    coeffs = linear_interpolation_coeffs(x, t=t, rectilinear=None)

    if t is None:
        t = torch.linspace(0, coeffs.size(-2) - 1, coeffs.size(-2), dtype=coeffs.dtype, device=coeffs.device)

    # Linear derivs
    derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1]).unsqueeze(-1)

    # Use the above to compute hermite coeffs
    hermite_coeffs = _setup_hermite_cubic_coeffs_w_backward_differences(t, coeffs, derivs, device=coeffs.device)

    return hermite_coeffs

