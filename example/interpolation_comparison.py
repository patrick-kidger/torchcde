import bashplotlib.histogram
import time
import torch
import torchcde


length = 400
input_channels = 20
hidden_channels = 64
backward = False
hist = False


class Func(torch.nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_channels, 64)
        self.linear2 = torch.nn.Linear(64, hidden_channels * input_channels)
        self.nfe = 0
        self.ts = []

    def reset(self):
        self.nfe = 0
        self.ts = []

    def forward(self, t, z):
        self.nfe += 1
        self.ts.append(t.item())
        return self.linear2(self.linear1(z).relu()).tanh().view(hidden_channels, input_channels)


t = torch.linspace(0, 1, length)
x = (torch.randn(length, input_channels) * 0.1).cumsum(dim=0)
func = Func()
z0 = torch.randn(hidden_channels, requires_grad=True)


def run(name, X):
    start = time.time()
    u = torchcde.cdeint(X, func, z0, t[[0, -1]])
    nfe_forward = func.nfe
    ts = func.ts
    func.reset()
    if backward:
        u[-1].sum().backward()
        nfe_backward = func.nfe
        func.reset()
    else:
        nfe_backward = 0
    print('{} NFE: Forward: {} Backward: {} Timespan: {}'.format(name, nfe_forward, nfe_backward, time.time() - start))
    if hist:
        bashplotlib.histogram.plot_hist(ts, bincount=200, xlab=True, regular=True)


print('NFE = Number of function evaluations')

cubic_coeffs = torchcde.natural_cubic_spline_coeffs(t, x)
cubic_interp = torchcde.NaturalCubicSpline(t, cubic_coeffs)
run('Natural Cubic Splines', cubic_interp)

linear_coeffs = torchcde.linear_interpolation_coeffs(t, x)

linear_interp = torchcde.LinearInterpolation(t, linear_coeffs, reparameterise=False)
run('Linear interpolation w/o reparam', linear_interp)

linear_reparam_interp = torchcde.LinearInterpolation(t, linear_coeffs, reparameterise=True)
run('Linear interpolation w/ reparam', linear_reparam_interp)

linear_region_interp = torchcde.LinearInterpolation(t, linear_coeffs, reparameterise=False).multiple_region()
run('Multiple-region linear interpolation w/o reparam', linear_region_interp)
