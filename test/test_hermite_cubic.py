import torch
from torchcde import hermite_cubic_coefficients_with_backward_differences, CubicSpline


# Represents a random Hermite cubic spline with unit time jumps
class _HermiteUnitTime:
    def __init__(self, data):
        x_next = data[..., 1:, :]
        x_prev = data[..., :-1, :]
        derivs_next = x_next - x_prev
        derivs_prev = torch.cat([derivs_next[..., [0], :], derivs_next[..., :-1, :]], axis=-2)
        self._a = x_prev
        self._b = derivs_prev
        self._two_c = 2 * 2 * (derivs_next - derivs_prev)
        self._three_d = -3 * (derivs_next - derivs_prev)

    def evaluate(self, fractional_part, index):
        fractional_part = fractional_part.unsqueeze(-1)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part


def test_hermite_cubic_unit_time():
    for num_channels in (1, 3, 6):
        for batch_dims in ((1,), (2, 3)):
            for length in (2, 5, 10):
                data = torch.randn(*batch_dims, length, num_channels, dtype=torch.float64)
                # Hermite
                hermite_coeffs = hermite_cubic_coefficients_with_backward_differences(data)
                spline = CubicSpline(hermite_coeffs)
                # Hermite with unit time
                hermite_cubic_unit = _HermiteUnitTime(data)
                # Test close
                times = torch.linspace(0, length, 10)
                for time in times:
                    fractional_part, index = spline._interpret_t(time)
                    assert torch.allclose(spline.evaluate(time), hermite_cubic_unit.evaluate(fractional_part, index))
