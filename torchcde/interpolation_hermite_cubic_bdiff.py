import torch
from torchcde import interpolation_base
from torchcde.interpolation_linear import linear_interpolation_coeffs


def _setup_hermite_cubic_coeffs_w_backward_differences(times, coeffs, derivs, device=None):
    """Compute the coefficients to smooth the linear interpolation."""
    x_prev = coeffs[..., :-1, :]
    x_next = coeffs[..., 1:, :]
    # Let x_0 - x_{-1} = x_1 - x_0
    derivs_prev = torch.cat((derivs[..., [0], :], derivs[..., :-1, :]), axis=-2)
    derivs_next = derivs
    x_diff = x_next - x_prev
    t_diff = (times[1:] - times[:-1]).unsqueeze(-1)
    # Coeffs
    D = x_prev
    C = derivs_prev
    B = (1 / t_diff ** 2) * (3 * (x_diff - C * t_diff) - t_diff * (derivs_next - derivs_prev))
    A = (1 / (3 * t_diff ** 2)) * (derivs_next - C - 2 * B * t_diff)
    matching_coeffs = torch.stack([A, B, C, D]).permute(1, 2, 3, 0)
    return matching_coeffs.to(device)


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

    # Linear derivs
    derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1]).unsqueeze(-1)

    # Use the above to compute hermite coeffs
    hermite_coeffs = _setup_hermite_cubic_coeffs_w_backward_differences(t, coeffs, derivs, device=coeffs.device)

    return hermite_coeffs


class HermiteCubicSplinesWithBackwardDifferences(interpolation_base.InterpolationBase):
    """Calculates the cubic Hermite spline interpolation with backwards differences and its derivative."""

    def __init__(self, coeffs, t=None):
        """
        Arguments:
            coeffs: As returned by hermite_cubic_coefficients_with_backward_differences.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
        """
        super(HermiteCubicSplinesWithBackwardDifferences, self).__init__()

        if t is None:
            t = torch.linspace(0, coeffs.size(-2) - 1, coeffs.size(-2), dtype=coeffs.dtype, device=coeffs.device)

        self.register_buffer("_t", t)
        self.register_buffer("_coeffs", coeffs)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def __len__(self):
        return len(self.grid_points)

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._derivs.dtype, device=self._derivs.device)
        maxlen = self._derivs.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    @staticmethod
    def _evaluate_matching_region(matching_coeffs, t, derivative=False):
        """Evaluates the evaluation/derivative polynomials for the Hermite matching regions."""
        device = matching_coeffs.device
        if derivative:
            t_powers = (
                torch.tensor([i * t ** (i - 1) for i in range(1, matching_coeffs.size(-1))]).flip(dims=[0]).to(device)
            )
            evaluation = (matching_coeffs[..., :-1] * t_powers).sum(dim=-1)
        else:
            t_powers = torch.cat([t ** i for i in range(matching_coeffs.size(-1))]).flip(dims=[0]).to(device)
            evaluation = (matching_coeffs * t_powers).sum(dim=-1)
        return evaluation

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        coeffs = self.hermite_coeffs[:, index]
        t_powers = torch.cat([fractional_part ** i for i in range(coeffs.size(-1))]).flip(dims=[0]).to(coeffs.device)
        evaluation = (coeffs * t_powers).sum(dim=-1)
        return evaluation

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        coeffs = self.hermite_coeffs[:, index]
        t_powers = (
            torch.tensor([i * fractional_part ** (i - 1) for i in range(1, coeffs.size(-1))])
            .flip(dims=[0])
            .to(coeffs.device)
        )
        deriv = (coeffs[..., :-1] * t_powers).sum(dim=-1)
        return deriv
