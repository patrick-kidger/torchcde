import torch
import torchcde


def test_shape():
    for method in ('rk4', 'dopri5'):
        for _ in range(10):
            num_points = torch.randint(low=5, high=100, size=(1,)).item()
            num_channels = torch.randint(low=1, high=3, size=(1,)).item()
            num_hidden_channels = torch.randint(low=1, high=5, size=(1,)).item()
            num_batch_dims = torch.randint(low=0, high=3, size=(1,)).item()
            batch_dims = []
            for _ in range(num_batch_dims):
                batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())

            values = torch.rand(*batch_dims, num_points, num_channels)

            coeffs = torchcde.natural_cubic_coeffs(values)
            spline = torchcde.NaturalCubicSpline(coeffs)

            class _Func(torch.nn.Module):
                def __init__(self):
                    super(_Func, self).__init__()
                    self.variable = torch.nn.Parameter(torch.rand(*[1 for _ in range(num_batch_dims)], 1, num_channels))

                def forward(self, t, z):
                    return z.sigmoid().unsqueeze(-1) + self.variable

            f = _Func()
            z0 = torch.rand(*batch_dims, num_hidden_channels)

            num_out_times = torch.randint(low=2, high=10, size=(1,)).item()
            start, end = spline.interval
            out_times = torch.rand(num_out_times, dtype=torch.float64).sort().values * (end - start) + start

            options = {}
            if method == 'rk4':
                options['step_size'] = 1. / num_points
            out = torchcde.cdeint(spline, f, z0, out_times, method=method, options=options, rtol=1e-4, atol=1e-6)
            assert out.shape == (*batch_dims, num_out_times, num_hidden_channels)


def test_tuple_input():
    xa = torch.rand(2, 10, 2)
    xb = torch.rand(10, 1)

    coeffs_a = torchcde.natural_cubic_coeffs(xa)
    coeffs_b = torchcde.natural_cubic_coeffs(xb)
    spline_a = torchcde.NaturalCubicSpline(coeffs_a)
    spline_b = torchcde.NaturalCubicSpline(coeffs_b)
    X = torchcde.TupleControl(spline_a, spline_b)

    def func(t, z):
        za, zb = z
        return za.sigmoid().unsqueeze(-1).repeat_interleave(2, dim=-1), zb.tanh().unsqueeze(-1)

    z0 = torch.rand(2, 3), torch.rand(5, requires_grad=True)
    out = torchcde.cdeint(X=X, func=func, z0=z0, t=X.interval, adjoint_params=())
    out[0].sum().backward()
    assert (z0[1].grad == 0).all()


def test_prod():
    x = torch.rand(2, 5, 1)
    X = torchcde.NaturalCubicSpline(torchcde.natural_cubic_coeffs(x))

    class F:
        def prod(self, t, z, dXdt):
            assert t.shape == ()
            assert z.shape == (2, 3)
            assert dXdt.shape == (2, 1)
            return -z * dXdt

    z0 = torch.rand(2, 3, requires_grad=True)
    out = torchcde.cdeint(X=X, func=F(), z0=z0, t=X.interval, adjoint_params=())
    out.sum().backward()
