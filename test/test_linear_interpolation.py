import torch
import torchcde


def test_random():
    def _points():
        yield 2
        yield 3
        yield 100
        for _ in range(10):
            yield torch.randint(low=2, high=100, size=(1,)).item()

    for reparameterise in ('none', 'bump'):
        for drop in (False, True):
            for num_points in _points():
                start = torch.rand(1).item() * 10 - 5
                end = torch.rand(1).item() * 10 - 5
                start, end = min(start, end), max(start, end)
                times = torch.linspace(start, end, num_points, dtype=torch.float64)
                num_channels = torch.randint(low=1, high=5, size=(1,)).item()
                m = torch.rand(num_channels, dtype=torch.float64) * 10 - 5
                c = torch.rand(num_channels, dtype=torch.float64) * 10 - 5
                values = m * times.unsqueeze(-1) + c

                values_clone = values.clone()
                if drop:
                    for values_slice in values_clone.unbind(dim=-1):
                        num_drop = int(num_points * torch.randint(low=1, high=4, size=(1,)).item() / 10)
                        num_drop = min(num_drop, num_points - 4)
                        to_drop = torch.randperm(num_points - 2)[:num_drop] + 1  # don't drop first or last
                        values_slice[to_drop] = float('nan')

                coeffs = torchcde.linear_interpolation_coeffs(times, values_clone)
                linear = torchcde.LinearInterpolation(times, coeffs, reparameterise)

                for time, value in zip(times, values):
                    linear_evaluate = linear.evaluate(time)
                    assert value.shape == linear_evaluate.shape
                    assert value.allclose(linear_evaluate, rtol=1e-4, atol=1e-6)
                    if reparameterise is False:
                        linear_derivative = linear.derivative(time)
                        assert m.shape == linear_derivative.shape
                        assert m.allclose(linear_derivative, rtol=1e-4, atol=1e-6)


def test_small():
    start = torch.rand(1).item() * 10 - 5
    end = torch.rand(1).item() * 10 - 5
    start, end = min(start, end), max(start, end)
    t = torch.tensor([start, end], dtype=torch.float64)
    x = torch.rand(2, 1, dtype=torch.float64)
    true_deriv = (x[1] - x[0]) / (end - start)
    coeffs = torchcde.linear_interpolation_coeffs(t, x)
    linear = torchcde.LinearInterpolation(t, coeffs)
    for time in torch.linspace(-1, 2, 100):
        true = x[0] + true_deriv * (time - t[0])
        pred = linear.evaluate(time)
        deriv = linear.derivative(time)
        assert true_deriv.shape == deriv.shape
        assert true_deriv.allclose(deriv)
        assert true.shape == pred.shape
        assert true.allclose(pred)


def test_specification_and_derivative():
    for reparameterise in ('none', 'bump'):
        for _ in range(10):
            for num_batch_dims in (0, 1, 2, 3):
                batch_dims = []
                for _ in range(num_batch_dims):
                    batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())
                length = torch.randint(low=5, high=10, size=(1,)).item()
                channels = torch.randint(low=1, high=5, size=(1,)).item()
                t = torch.linspace(0, 1, length, dtype=torch.float64)
                x = torch.rand(*batch_dims, length, channels, dtype=torch.float64)
                coeffs = torchcde.linear_interpolation_coeffs(t, x)
                spline = torchcde.LinearInterpolation(t, coeffs, reparameterise)
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
