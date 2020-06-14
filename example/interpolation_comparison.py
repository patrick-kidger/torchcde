######################
# Different interpolation schemes have different strengths and weaknesses.
# The two main ones we consider here are natural cubic splines and linear interpolation.
# A good interpolation scheme should satisify three conditions:
# - Be causal.
# - Depend in a sensible way on the data. (For example, quadratic splines can be unbounded on bounded data, so they're
#   not a good choice!)
# - Be smooth. This makes solving them with adaptive step size solvers much easier.
#
# Both linear interpolation and natural cubic splines depend on the data sensibly. (Both are bounded by some multiple of
# the norm of the data.)
# In terms of smoothness: 'naive' linear interpolation is very non-smooth; it has lots of jagged peaks. This is very
# expensive to solve. However it is possible to _reparameterise_ the path: produce a path which has the same image
# that is traced in the same order, but does so at a different speed. We can use this to slow down near each peak, to
# make them easier to resolve.
# Natural cubic splines are very smooth, and are easy to resolve.
# Linear interpolation is causal, and natural cubic splines are noncausal.
# Generally, we see that naive linear interpolation is about 750% [not a typo] more expensive than natural cubic
# splines, whilst reparameterised linear interpolation is about 45% more expensive than natural cubic splines. So if
# causality is not required then natural cubic splines are recommended. If causality is required, then reparameterised
# linear interpolation is preferred.
#
# Here we demonstrate the costs that have just been stated above.
######################

import torch
import torchcontroldiffeq


t = torch.linspace(0, 1, 1000)
x = (torch.randn(1000, 3) * 0.1).cumsum(dim=0)


class Func(torch.nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        self.variable = torch.nn.Parameter(torch.randn(1, 3))
        self.nfe = 0

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, z):
        self.nfe += 1
        return z.sigmoid().unsqueeze(-1) + self.variable


func = Func()
z0 = torch.randn(3, requires_grad=True)
t_ = torch.linspace(0, 1, 2)


def cost(X):
    u = torchcontroldiffeq.cdeint(X, func, z0, t_, atol=1e-5, rtol=1e-5)
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
coeffs = torchcontroldiffeq.natural_cubic_spline_coeffs(t, x)
cubic_interp = torchcontroldiffeq.NaturalCubicSpline(t, coeffs)
cubic_nfe_forward, cubic_nfe_backward = cost(cubic_interp)
print('Natural Cubic Splines NFE: Forward: {} Backward: {}'.format(cubic_nfe_forward, cubic_nfe_backward))


######################
# Linear interpolation
######################
coeffs = torchcontroldiffeq.linear_interpolation_coeffs(t, x)
linear_interp = torchcontroldiffeq.LinearInterpolation(t, coeffs, reparameterise=False)
linear_nfe_forward, linear_nfe_backward = cost(linear_interp)
print('Linear interpolation w/o reparam NFE: Forward: {} Backward: {}'.format(linear_nfe_forward, linear_nfe_backward))

######################
# Reparameterised linear interpolation
######################
coeffs = torchcontroldiffeq.linear_interpolation_coeffs(t, x)
linear_reparam_interp = torchcontroldiffeq.LinearInterpolation(t, coeffs, reparameterise=True)
linear_reparam_nfe_forward, linear_reparam_nfe_backward = cost(linear_reparam_interp)
print('Linear interpolation w/ reparam NFE: Forward: {} Backward: {}'.format(linear_reparam_nfe_forward,
                                                                              linear_reparam_nfe_backward))

linear_ratio = (linear_nfe_forward + linear_nfe_backward) / (cubic_nfe_forward + cubic_nfe_backward)
linear_reparam_ratio = (linear_reparam_nfe_forward + linear_reparam_nfe_backward) / (cubic_nfe_forward +
                                                                                     cubic_nfe_backward)

print()
print('Linear interpolation is {:.3f}% slower than natural cubic splines.'.format((linear_ratio - 1) * 100))
print('Reparameterised linear interpolation is {:.3f}% slower than natural cubic splines.'
      .format((linear_reparam_ratio - 1) * 100))
