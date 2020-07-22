import torch
import torchcde


t = torch.linspace(0, 1, 1000)
x = (torch.randn(1000, 3) * 0.1).cumsum(dim=0)


class Func(torch.nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        self.variable = torch.nn.Parameter(torch.randn(1, 3))
        self.nfe = 0

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        return z.sigmoid().unsqueeze(-1) + self.variable


func = Func()
z0 = torch.randn(3, requires_grad=True)
t_ = torch.linspace(0, 1, 2)


def cost(X):
    u = torchcde.cdeint(X, func, z0, t_, atol=1e-5, rtol=1e-5)
    nfe_forward = func.nfe
    func.reset_nfe()
    u[-1].sum().backward()
    nfe_backward = func.nfe
    func.reset_nfe()
    return nfe_forward, nfe_backward


print('NFE = Number of function evaluations')


######################
# Natural cubic splines
######################
coeffs = torchcde.natural_cubic_spline_coeffs(t, x)
cubic_interp = torchcde.NaturalCubicSpline(t, coeffs)
cubic_nfe_forward, cubic_nfe_backward = cost(cubic_interp)
print('Natural Cubic Splines NFE: Forward: {} Backward: {}'.format(cubic_nfe_forward, cubic_nfe_backward))


######################
# Linear interpolation
######################
coeffs = torchcde.linear_interpolation_coeffs(t, x)
linear_interp = torchcde.LinearInterpolation(t, coeffs, reparameterise=False)
linear_nfe_forward, linear_nfe_backward = cost(linear_interp)
print('Linear interpolation w/o reparam NFE: Forward: {} Backward: {}'.format(linear_nfe_forward, linear_nfe_backward))

######################
# Reparameterised linear interpolation
######################
coeffs = torchcde.linear_interpolation_coeffs(t, x)
linear_reparam_interp = torchcde.LinearInterpolation(t, coeffs, reparameterise=True)
linear_reparam_nfe_forward, linear_reparam_nfe_backward = cost(linear_reparam_interp)
print('Linear interpolation w/ reparam NFE: Forward: {} Backward: {}'.format(linear_reparam_nfe_forward,
                                                                             linear_reparam_nfe_backward))


######################
# Multiple-region inear interpolation
######################
coeffs = torchcde.linear_interpolation_coeffs(t, x)
linear_interp = torchcde.LinearInterpolation(t, coeffs, reparameterise=False).multiple_region()
linear_nfe_forward, linear_nfe_backward = cost(linear_interp)
print('Multiple-region linear interpolation w/o reparam NFE: Forward: {} Backward: {}'.format(linear_nfe_forward,
                                                                                              linear_nfe_backward))
