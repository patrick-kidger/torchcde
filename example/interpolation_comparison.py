import time
import torch
import torchcde


length = 400
input_channels = 20
hidden_channels = 64


class Func(torch.nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_channels, 64)
        self.linear2 = torch.nn.Linear(64, hidden_channels * input_channels)
        self.nfe = 0

    def reset(self):
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        return self.linear2(self.linear1(z).relu()).tanh().view(hidden_channels, input_channels)


t = torch.linspace(0, 1, length)
x = (torch.randn(length, input_channels) * 0.1).cumsum(dim=0)
func = Func()
z0 = torch.randn(hidden_channels, requires_grad=True)


def run(name, X, **kwargs):
    start = time.time()
    u = torchcde.cdeint(X, func, z0, t[[0, -1]], method='dopri5', options=kwargs)
    nfe_forward = func.nfe
    func.reset()
    u[-1].sum().backward()
    nfe_backward = func.nfe
    func.reset()
    timespan = time.time() - start
    print('{}: NFE Forward: {}, NFE Backward: {}, Timespan: {}'.format(name, nfe_forward, nfe_backward, timespan))


print('NFE = Number of function evaluations')

cubic_coeffs = torchcde.natural_cubic_spline_coeffs(t, x)
linear_coeffs = torchcde.linear_interpolation_coeffs(t, x)

print("\nCubic splines do pretty well!")
cubic_X = torchcde.NaturalCubicSpline(t, cubic_coeffs)
run('Natural Cubic Splines', cubic_X)

print("\nNaive linear interpolation takes a very long time.")
linear_X = torchcde.LinearInterpolation(t, linear_coeffs)
run('Linear interpolation w/o reparam', linear_X)

print("\nReparameterising it helps...")
linear_reparam_X = torchcde.LinearInterpolation(t, linear_coeffs, reparameterise=True)
run('Linear interpolation w/ reparam', linear_reparam_X)

print("\n...but an even better choice is to tell the solver about the jumps!")
linear_grid_X = torchcde.LinearInterpolation(t, linear_coeffs)
run('Grid-aware linear interpolation eps=1e-5', linear_grid_X, grid_points=t, eps=1e-5)

print("\nDon't forget the `eps` argument. If using linear interpolation then you should always set this to a small "
      "number above zero, like 1e-5. (And making it bigger or smaller won't really change anything.)")
linear_grid_zero_eps_X = torchcde.LinearInterpolation(t, linear_coeffs)
run('Grid-aware linear interpolation eps=0', linear_grid_zero_eps_X, grid_points=t)
