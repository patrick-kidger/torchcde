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
                    linear = torchcde.LinearInterpolation(coeffs, t=t_, reparameterise=reparameterise)

                    for time, value in zip(t, values):
                        linear_evaluate = linear.evaluate(time)
                        assert value.shape == linear_evaluate.shape
                        assert value.allclose(linear_evaluate, rtol=1e-4, atol=1e-6)
                        if reparameterise is False:
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
        for reparameterise in ('none', 'bump'):
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
                    spline = torchcde.LinearInterpolation(coeffs, t=t_, reparameterise=reparameterise)
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
    # Some fucntions
    pad_sequence = lambda sequences: torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=nan)
    ffill = torchcde.misc.torch_ffill

    # Simple test
    nan = float('nan')
    t1 = torch.Tensor([0.1, 0.2, 0.9]).view(-1, 1)
    t2 = torch.Tensor([0.2, 0.3]).view(-1, 1)
    x1 = torch.Tensor([0.4, nan, 1.1]).view(-1, 1)
    x2 = torch.Tensor([nan, 2.]).view(-1, 1)
    x = pad_sequence([torch.cat((t1, x1), -1), torch.cat((t2, x2), -1)])
    x1_true = torch.Tensor([[0.1, 0.2, 0.2, 0.9, 0.9], [0.4, 0.4, 0.4, 0.4, 1.1]]).T.view(-1, 2)
    x2_true = torch.Tensor([[0.2, 0.3, 0.3], [nan, nan, 2.]]).T.view(-1, 2)
    rect_true = ffill(pad_sequence([x1_true, x2_true]))
    rectilinear = torchcde.interpolation_linear._prepare_rectilinear_interpolation(x, 0)
    assert rect_true.allclose(rectilinear, equal_nan=True)
    # Test also if we swap time time dimension
    x_swap = x[:, :, [1, 0]]
    rectilinear_swap = torchcde.interpolation_linear._prepare_rectilinear_interpolation(x_swap, 1)
    assert ffill(rect_true[:, :, [1, 0]]).allclose(rectilinear_swap, equal_nan=True)

    n_channels = 10
    for _ in range(5):
        # Build some data with time
        t_starts = torch.randn(5) ** 2
        ts = [torch.linspace(t_start, t_start + 10, torch.randint(0, 50, (1,)).item()) for t_start in t_starts]
        xs = [torch.randn(len(t), n_channels - 1) for t in ts]
        x = pad_sequence([torch.cat([t_.view(-1, 1), x_], 1) for t_, x_ in zip(ts, xs)])
        # Add some random nans about the place
        mask = torch.randint(0, 5, (x.size(0), x.size(1), x.size(2) - 1)).to(float)
        mask[mask == 0] = float('nan')
        x[:, :, 1:] = x[:, :, 1:] * mask
        # Fill
        x_ffilled = ffill(x)
        # Compute the true solution
        N, L, C = x_ffilled.shape
        rect_true = torch.zeros(N, 2 * L - 1, C)
        lag = torch.cat([x_ffilled[:, 1:, [0]], x_ffilled[:, :-1, 1:]], -1)
        rect_true[:, ::2, ] = x_ffilled
        rect_true[:, 1::2] = lag
        # Rectilinear solution
        rectilinear = torchcde.interpolation_linear._prepare_rectilinear_interpolation(x, 0)
        assert rect_true.allclose(rectilinear, equal_nan=True)