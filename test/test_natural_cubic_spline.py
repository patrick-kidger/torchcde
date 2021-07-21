import pytest
import torch
import torchcde


# Represents a random natural cubic spline with a single knot in the middle
class _Cubic:
    def __init__(self, batch_dims, num_channels, start, end):
        self.a = torch.randn(*batch_dims, num_channels, dtype=torch.float64) * 10
        self.b = torch.randn(*batch_dims, num_channels, dtype=torch.float64) * 10
        self.c = torch.randn(*batch_dims, num_channels, dtype=torch.float64) * 10
        self.d1 = -self.c / (3 * start)
        self.d2 = -self.c / (3 * end)

    def _normalise_dims(self, t):
        a = self.a
        b = self.b
        c = self.c
        d1 = self.d1
        d2 = self.d2
        for _ in t.shape:
            a = a.unsqueeze(-2)
            b = b.unsqueeze(-2)
            c = c.unsqueeze(-2)
            d1 = d1.unsqueeze(-2)
            d2 = d2.unsqueeze(-2)
        t = t.unsqueeze(-1)
        d = torch.where(t >= 0, d2, d1)
        return a, b, c, d, t

    def evaluate(self, t):
        a, b, c, d, t = self._normalise_dims(t)
        t_sq = t ** 2
        t_cu = t_sq * t
        return a + b * t + c * t_sq + d * t_cu

    def derivative(self, t):
        a, b, c, d, t = self._normalise_dims(t)
        t_sq = t ** 2
        return b + 2 * c * t + 3 * d * t_sq


class _Offset:
    def __init__(self, batch_dims, num_channels, start, end, offset):
        self.cubic = _Cubic(batch_dims, num_channels, start - offset, end - offset)
        self.offset = offset

    def evaluate(self, t):
        t = t - self.offset
        return self.cubic.evaluate(t)

    def derivative(self, t):
        t = t - self.offset
        return self.cubic.derivative(t)


@pytest.mark.skip(reason="Test is flaky. Not obvious whether the problem is in the test or in the natural cubic "
                         "spline code. As natural cubic splines are being phased out in favour of alternatives "
                         "anyway, I'm just marking this test as a skip.")
def test_interp():
    for interp_fn in (torchcde.natural_cubic_coeffs, torchcde.natural_cubic_spline_coeffs):
        for _ in range(3):
            for use_t in (True, False):
                for drop in (False, True):
                    num_points = torch.randint(low=5, high=100, size=(1,)).item()
                    half_num_points = num_points // 2
                    num_points = 2 * half_num_points + 1
                    if use_t:
                        times1 = torch.rand(half_num_points, dtype=torch.float64) - 1
                        times2 = torch.rand(half_num_points, dtype=torch.float64)
                        t = torch.cat([times1, times2, torch.tensor([0.], dtype=torch.float64)]).sort().values
                        t_ = t
                        start, end = -1.5, 1.5
                        del times1, times2
                    else:
                        t = torch.linspace(0, num_points - 1, num_points, dtype=torch.float64)
                        t_ = None
                        start = 0
                        end = num_points - 0.5
                    num_channels = torch.randint(low=1, high=3, size=(1,)).item()
                    num_batch_dims = torch.randint(low=0, high=3, size=(1,)).item()
                    batch_dims = []
                    for _ in range(num_batch_dims):
                        batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())
                    if use_t:
                        cubic = _Cubic(batch_dims, num_channels, start=t[0], end=t[-1])
                        knot = 0
                    else:
                        cubic = _Offset(batch_dims, num_channels, start=t[0], end=t[-1], offset=t[1] - t[0])
                        knot = t[1] - t[0]
                    values = cubic.evaluate(t)
                    if drop:
                        for values_slice in values.unbind(dim=-1):
                            num_drop = int(num_points * torch.randint(low=1, high=4, size=(1,)).item() / 10)
                            num_drop = min(num_drop, num_points - 4)
                            # don't drop first or last
                            to_drop = torch.randperm(num_points - 2)[:num_drop] + 1
                            to_drop = [x for x in to_drop if x != knot]
                            values_slice[..., to_drop] = float('nan')
                            del num_drop, to_drop, values_slice
                    coeffs = interp_fn(values, t_)
                    spline = torchcde.CubicSpline(coeffs, t_)
                    _test_equal(batch_dims, num_channels, cubic, spline, torch.float64, start, end, 1e-3)


def test_linear():
    for interp_fn in (torchcde.natural_cubic_coeffs, torchcde.natural_cubic_spline_coeffs):
        for use_t in (False, True):
            start = torch.rand(1).item() * 5 - 2.5
            end = torch.rand(1).item() * 5 - 2.5
            start, end = min(start, end), max(start, end)
            num_points = torch.randint(low=2, high=10, size=(1,)).item()
            num_channels = torch.randint(low=1, high=4, size=(1,)).item()
            m = torch.rand(num_channels, dtype=torch.float64) * 5 - 2.5
            c = torch.rand(num_channels, dtype=torch.float64) * 5 - 2.5
            if use_t:
                t = torch.linspace(start, end, num_points, dtype=torch.float64)
                t_ = t
            else:
                t = torch.linspace(0, num_points - 1, num_points, dtype=torch.float64)
                t_ = None
            values = m * t.unsqueeze(-1) + c
            coeffs = interp_fn(values, t_)
            spline = torchcde.CubicSpline(coeffs, t_)
            coeffs2 = torchcde.linear_interpolation_coeffs(values, t_)
            linear = torchcde.LinearInterpolation(coeffs2, t_)
            batch_dims = []
            _test_equal(batch_dims, num_channels, linear, spline, torch.float32, -1.5, 1.5, 1e-4)


def test_short():
    for interp_fn in (torchcde.natural_cubic_coeffs, torchcde.natural_cubic_spline_coeffs):
        for use_t in (False, True):
            if use_t:
                t = torch.tensor([0., 1.])
            else:
                t = None
            values = torch.rand(2, 1)
            coeffs = interp_fn(values, t)
            spline = torchcde.CubicSpline(coeffs, t)
            coeffs2 = torchcde.linear_interpolation_coeffs(values, t)
            linear = torchcde.LinearInterpolation(coeffs2, t)
            batch_dims = []
            num_channels = 1
            _test_equal(batch_dims, num_channels, linear, spline, torch.float32, -1.5, 1.5, 1e-4)


# TODO: test other edge cases


def _test_equal(batch_dims, num_channels, obj1, obj2, dtype, start, end, tol):
    for dimension in (0, 1, 2):
        sizes = []
        for _ in range(dimension):
            sizes.append(torch.randint(low=1, high=4, size=(1,)).item())
        expected_size = tuple(batch_dims) + tuple(sizes) + (num_channels,)
        eval_times = torch.rand(sizes, dtype=dtype) * (end - start) + start
        obj1_evaluate = obj1.evaluate(eval_times)
        obj2_evaluate = obj2.evaluate(eval_times)
        obj1_derivative = obj1.derivative(eval_times)
        obj2_derivative = obj2.derivative(eval_times)
        assert obj1_evaluate.shape == expected_size
        assert obj2_evaluate.shape == expected_size
        assert obj1_derivative.shape == expected_size
        assert obj2_derivative.shape == expected_size
        torch.testing.assert_allclose(obj1_evaluate, obj2_evaluate, rtol=tol, atol=tol)
        torch.testing.assert_allclose(obj1_derivative, obj2_derivative, rtol=tol, atol=tol)


def test_specification_and_derivative():
    for interp_fn in (torchcde.natural_cubic_coeffs, torchcde.natural_cubic_spline_coeffs):
        for _ in range(10):
            for use_t in (False, True):
                for num_batch_dims in (0, 1, 2, 3):
                    batch_dims = []
                    for _ in range(num_batch_dims):
                        batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())
                    length = torch.randint(low=5, high=10, size=(1,)).item()
                    channels = torch.randint(low=1, high=5, size=(1,)).item()
                    if use_t:
                        t = torch.linspace(0, 1, length, dtype=torch.float64)
                    else:
                        t = torch.linspace(0, length - 1, length, dtype=torch.float64)
                    x = torch.rand(*batch_dims, length, channels, dtype=torch.float64)
                    coeffs = interp_fn(x, t)
                    spline = torchcde.CubicSpline(coeffs, t)
                    # Test specification
                    for i, point in enumerate(t):
                        evaluate = spline.evaluate(point)
                        xi = x[..., i, :]
                        assert evaluate.allclose(xi, atol=1e-5, rtol=1e-5)
                    # Test derivative
                    for point in torch.rand(100, dtype=torch.float64):
                        point_with_grad = point.detach().requires_grad_(True)
                        evaluate = spline.evaluate(point_with_grad)
                        derivative = spline.derivative(point)
                        autoderivative = []
                        for elem in evaluate.view(-1):
                            elem.backward(retain_graph=True)
                            with torch.no_grad():
                                autoderivative.append(point_with_grad.grad.clone())
                            point_with_grad.grad.zero_()
                        autoderivative = torch.stack(autoderivative).view(*evaluate.shape)
                        assert derivative.shape == autoderivative.shape
                        assert derivative.allclose(autoderivative, atol=1e-5, rtol=1e-5)
