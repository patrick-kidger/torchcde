import torch
import torchcde
import pytest


def test_random():
    def _points():
        yield 2
        yield 3
        yield 100
        for _ in range(10):
            yield torch.randint(low=2, high=100, size=(1,)).item()

    for drop in (False, True):
        for use_t in (False, True):
            for num_points in _points():
                if use_t:
                    start = torch.rand(1).item() * 10 - 5
                    end = torch.rand(1).item() * 10 - 5
                    start, end = min(start, end), max(start, end)
                    t = torch.linspace(start, end, num_points, dtype=torch.float64)
                    t_ = t
                else:
                    t = torch.linspace(0, num_points - 1, num_points, dtype=torch.float64)
                    t_ = None
                num_channels = torch.randint(low=1, high=5, size=(1,)).item()
                m = torch.rand(num_channels, dtype=torch.float64) * 10 - 5
                c = torch.rand(num_channels, dtype=torch.float64) * 10 - 5
                values = m * t.unsqueeze(-1) + c

                values_clone = values.clone()
                if drop:
                    for values_slice in values_clone.unbind(dim=-1):
                        num_drop = int(num_points * torch.randint(low=1, high=4, size=(1,)).item() / 10)
                        num_drop = min(num_drop, num_points - 4)
                        to_drop = torch.randperm(num_points - 2)[:num_drop] + 1  # don't drop first or last
                        values_slice[to_drop] = float('nan')

                coeffs = torchcde.linear_interpolation_coeffs(values_clone, t=t_)
                linear = torchcde.LinearInterpolation(coeffs, t=t_)

                for time, value in zip(t, values):
                    linear_evaluate = linear.evaluate(time)
                    assert value.shape == linear_evaluate.shape
                    assert value.allclose(linear_evaluate, rtol=1e-4, atol=1e-6)
                    linear_derivative = linear.derivative(time)
                    assert m.shape == linear_derivative.shape
                    assert m.allclose(linear_derivative, rtol=1e-4, atol=1e-6)


def test_small():
    for use_t in (False, True):
        if use_t:
            start = torch.rand(1).item() * 10 - 5
            end = torch.rand(1).item() * 10 - 5
            start, end = min(start, end), max(start, end)
            t = torch.tensor([start, end], dtype=torch.float64)
            t_ = t
        else:
            start = 0
            end = 1
            t = torch.tensor([0., 1.], dtype=torch.float64)
            t_ = None
        x = torch.rand(2, 1, dtype=torch.float64)
        true_deriv = (x[1] - x[0]) / (end - start)
        coeffs = torchcde.linear_interpolation_coeffs(x, t=t_)
        linear = torchcde.LinearInterpolation(coeffs, t=t_)
        for time in torch.linspace(-1, 2, 100):
            true = x[0] + true_deriv * (time - t[0])
            pred = linear.evaluate(time)
            deriv = linear.derivative(time)
            assert true_deriv.shape == deriv.shape
            assert true_deriv.allclose(deriv)
            assert true.shape == pred.shape
            assert true.allclose(pred)


def test_specification_and_derivative():
    for use_t in (False, True):
        for _ in range(10):
            for num_batch_dims in (0, 1, 2, 3):
                batch_dims = []
                for _ in range(num_batch_dims):
                    batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())
                length = torch.randint(low=5, high=10, size=(1,)).item()
                channels = torch.randint(low=1, high=5, size=(1,)).item()
                if use_t:
                    t = torch.linspace(0, 1, length, dtype=torch.float64)
                    t_ = t
                else:
                    t = torch.linspace(0, length - 1, length, dtype=torch.float64)
                    t_ = None
                x = torch.rand(*batch_dims, length, channels, dtype=torch.float64)
                coeffs = torchcde.linear_interpolation_coeffs(x, t=t_)
                spline = torchcde.LinearInterpolation(coeffs, t=t_)
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


def test_rectilinear_preparation():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        # Simple test
        nan = float('nan')
        t1 = torch.tensor([0.1, 0.2, 0.9]).view(-1, 1).to(device)
        t2 = torch.tensor([0.2, 0.3]).view(-1, 1).to(device)
        x1 = torch.tensor([0.4, nan, 1.1]).view(-1, 1).to(device)
        x2 = torch.tensor([nan, 2.]).view(-1, 1).to(device)
        x = torch.nn.utils.rnn.pad_sequence(
            [torch.cat((t1, x1), -1), torch.cat((t2, x2), -1)], batch_first=True, padding_value=nan
        )
        # We have to fill the time index forward because we currently dont allow nan times for rectilinear
        x[:, :, 0] = torchcde.misc.forward_fill(x[:, :, 0], fill_index=-1)
        # Build true solution
        x1_true = torch.tensor([[0.1, 0.2, 0.2, 0.9, 0.9], [0.4, 0.4, 0.4, 0.4, 1.1]]).T.view(-1, 2).to(device)
        x2_true = torch.tensor([[0.2, 0.3, 0.3, 0.3, 0.3], [2., 2., 2., 2., 2.]]).T.view(-1, 2).to(device)
        rect_true = torch.stack((x1_true, x2_true))
        # Apply rectilinear and compare
        rectilinear = torchcde.linear_interpolation_coeffs(x, rectilinear=0)
        assert torch.equal(rect_true[~torch.isnan(rect_true)], rectilinear[~torch.isnan(rectilinear)])
        # Test also if we swap time time dimension
        x_swap = x[:, :, [1, 0]]
        rectilinear_swap = torchcde.linear_interpolation_coeffs(x_swap, rectilinear=1)
        rect_swp = rect_true[:, :, [1, 0]]
        assert torch.equal(rect_swp, rectilinear_swap)

        # Additionally try a 2d case
        assert torch.equal(rect_true[0], torchcde.linear_interpolation_coeffs(x[0], rectilinear=0))
        # And a 4d case
        x_4d = torch.stack([x, x])
        rect_true_4d = torch.stack([rect_true, rect_true])
        assert torch.equal(rect_true_4d, torchcde.linear_interpolation_coeffs(x_4d, rectilinear=0))

        # Ensure error is thrown if time has a nan value anywhere
        x_time_nan = x.clone()
        x_time_nan[0, 1, 0] = float('nan')
        pytest.raises(AssertionError, torchcde.linear_interpolation_coeffs, x_time_nan, rectilinear=0)

        # Some randoms tests
        for _ in range(5):
            # Build some data with time
            t_starts = torch.randn(5).to(device) ** 2
            ts = [torch.linspace(s, s + 10, torch.randint(2, 50, (1,)).item()).to(device) for s in t_starts]
            xs = [torch.randn(len(t), 10 - 1).to(device) for t in ts]
            x = torch.nn.utils.rnn.pad_sequence(
                [torch.cat([t_.view(-1, 1), x_], dim=1) for t_, x_ in zip(ts, xs)], batch_first=True, padding_value=nan
            )
            # Add some random nans about the place
            mask = torch.randint(0, 5, (x.size(0), x.size(1), x.size(2) - 1), dtype=torch.float).to(device)
            mask[mask == 0] = float('nan')
            x[:, :, 1:] = x[:, :, 1:] * mask
            # We have to fill the time index forward because we currently dont allow nan times for rectilinear
            x[:, :, 0] = torchcde.misc.forward_fill(x[:, :, 0], fill_index=-1)
            # Fill
            x_ffilled = torchcde.misc.forward_fill(x)
            # Compute the true solution
            N, L, C = x_ffilled.shape
            rect_true = torch.zeros(N, 2 * L - 1, C).to(device)
            lag = torch.cat([x_ffilled[:, 1:, [0]], x_ffilled[:, :-1, 1:]], dim=-1)
            rect_true[:, ::2, ] = x_ffilled
            rect_true[:, 1::2] = lag
            # Need to backfill rect true
            # Rectilinear solution
            rectilinear = torchcde.linear_interpolation_coeffs(x, rectilinear=0)
            assert torch.equal(rect_true[~torch.isnan(rect_true)], rectilinear[~torch.isnan(rect_true)])
