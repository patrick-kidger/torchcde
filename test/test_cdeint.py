import torch
import torchcontroldiffeq


def test_shape():
    for _ in range(10):
        num_points = torch.randint(low=5, high=100, size=(1,)).item()
        num_channels = torch.randint(low=1, high=3, size=(1,)).item()
        num_hidden_channels = torch.randint(low=1, high=5, size=(1,)).item()
        num_batch_dims = torch.randint(low=0, high=3, size=(1,)).item()
        batch_dims = []
        for _ in range(num_batch_dims):
            batch_dims.append(torch.randint(low=1, high=3, size=(1,)).item())

        times = torch.rand(num_points)
        values = torch.rand(*batch_dims, num_points, num_channels)

        coeffs = torchcontroldiffeq.natural_cubic_spline_coeffs(times, values)
        spline = torchcontroldiffeq.NaturalCubicSpline(times, coeffs)

        class _Func(torch.nn.Module):
            def __init__(self):
                super(_Func, self).__init__()
                self.variable = torch.nn.Parameter(torch.rand(*[1 for _ in range(num_batch_dims)], 1, num_channels))

            def forward(self, z):
                return z.sigmoid().unsqueeze(-1) + self.variable

        f = _Func()
        z0 = torch.rand(*batch_dims, num_hidden_channels)

        num_out_times = torch.randint(low=2, high=10, size=(1,)).item()
        out_times = torch.rand(num_out_times, dtype=torch.float64).sort().values * (times[-1] - times[0]) + times[0]

        out = torchcontroldiffeq.cdeint(spline, f, z0, out_times, atol=0.001, rtol=0.001)
        assert out.shape == (*batch_dims, num_out_times, num_hidden_channels)
