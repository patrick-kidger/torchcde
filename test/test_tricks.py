import pytest
import torch
import torchcde


class _Func(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(_Func, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.variable = torch.nn.Parameter(torch.rand(1, 1, input_size))

    def forward(self, t, z):
        assert z.shape == (1, self.hidden_size)
        out = z.sigmoid().unsqueeze(-1) + self.variable
        assert out.shape == (1, self.hidden_size, self.input_size)
        return out


# Test that gradients can propagate through the controlling path at all
def test_grad_paths():
    for method in ('rk4', 'dopri5'):
        for adjoint in (True, False):
            t = torch.linspace(0, 9, 10, requires_grad=True)
            path = torch.rand(1, 10, 3, requires_grad=True)
            coeffs = torchcde.natural_cubic_coeffs(path, t)
            cubic_spline = torchcde.CubicSpline(coeffs, t)
            z0 = torch.rand(1, 3, requires_grad=True)
            func = _Func(input_size=3, hidden_size=3)
            t_ = torch.tensor([0., 9.], requires_grad=True)

            if adjoint:
                kwargs = dict(adjoint_params=tuple(func.parameters()) + (coeffs, t))
            else:
                kwargs = {}
            z = torchcde.cdeint(X=cubic_spline, func=func, z0=z0, t=t_, adjoint=adjoint, method=method, rtol=1e-4,
                                atol=1e-6, **kwargs)
            assert z.shape == (1, 2, 3)
            assert t.grad is None
            assert path.grad is None
            assert z0.grad is None
            assert func.variable.grad is None
            assert t_.grad is None
            z[:, 1].sum().backward()
            assert isinstance(t.grad, torch.Tensor)
            assert isinstance(path.grad, torch.Tensor)
            assert isinstance(z0.grad, torch.Tensor)
            assert isinstance(func.variable.grad, torch.Tensor)
            assert isinstance(t_.grad, torch.Tensor)


# Test that gradients flow back through multiple CDEs stacked on top of one another, and that they do so correctly
# without going through earlier parts of the graph multiple times.
def test_stacked_paths():
    class Record(torch.autograd.Function):
        @staticmethod
        def forward(ctx, name, x):
            ctx.name = name
            return x

        @staticmethod
        def backward(ctx, x):
            if hasattr(ctx, 'been_here_before'):
                pytest.fail(ctx.name)
            ctx.been_here_before = True
            return None, x

    coeff_paths = [(torchcde.linear_interpolation_coeffs, torchcde.LinearInterpolation),
                   (torchcde.natural_cubic_coeffs, torchcde.CubicSpline)]
    for adjoint in (False, True):
        for first_coeffs, First in coeff_paths:
            for second_coeffs, Second in coeff_paths:
                first_path = torch.rand(1, 1000, 2, requires_grad=True)
                first_coeff = first_coeffs(first_path)
                first_X = First(first_coeff)
                first_func = _Func(input_size=2, hidden_size=2)

                second_t = torch.linspace(0, 999, 100)
                if adjoint:
                    kwargs = dict(adjoint_params=tuple(first_func.parameters()) + (first_coeff,))
                else:
                    kwargs = {}
                second_path = torchcde.cdeint(X=first_X, func=first_func, z0=torch.rand(1, 2),
                                              t=second_t, adjoint=adjoint, method='rk4', options=dict(step_size=10),
                                              **kwargs)
                second_path = Record.apply('second', second_path)
                second_coeff = second_coeffs(second_path, second_t)
                second_X = Second(second_coeff, second_t)
                second_func = _Func(input_size=2, hidden_size=2)

                third_t = torch.linspace(0, 999, 10)
                if adjoint:
                    kwargs = dict(adjoint_params=tuple(second_func.parameters()) + (second_coeff, second_t))
                else:
                    kwargs = {}
                third_path = torchcde.cdeint(X=second_X, func=second_func, z0=torch.rand(1, 2),
                                             t=third_t, adjoint=adjoint, method='rk4', options=dict(step_size=10),
                                             **kwargs)
                third_path = Record.apply('third', third_path)
                assert first_func.variable.grad is None
                assert second_func.variable.grad is None
                assert first_path.grad is None
                third_path[:, -1].sum().backward()
                assert isinstance(second_func.variable.grad, torch.Tensor)
                assert isinstance(first_func.variable.grad, torch.Tensor)
                assert isinstance(first_path.grad, torch.Tensor)


# Tests that the trick in which we use detaches in the backward pass if possible, does in fact work.
# It's a bit superfluous to test it here now that we've upstreamed it into torchdiffeq, but oh well.
def test_detach_trick():
    path = torch.rand(1, 10, 3)
    interp = torchcde.CubicSpline(torchcde.natural_cubic_coeffs(path))

    func = _Func(input_size=3, hidden_size=3)

    for adjoint in (True, False):
        variable_grads = []
        z0 = torch.rand(1, 3)
        for t_grad in (True, False):
            t_ = torch.tensor([0., 9.], requires_grad=t_grad)
            # Don't test dopri5. We will get different results then, because the t variable will force smaller step
            # sizes and thus slightly different results.
            z = torchcde.cdeint(X=interp, z0=z0, func=func, t=t_, adjoint=adjoint, method='rk4',
                                options=dict(step_size=0.5))
            z[:, -1].sum().backward()
            variable_grads.append(func.variable.grad.clone())
            func.variable.grad.zero_()

        for elem in variable_grads[1:]:
            assert (elem == variable_grads[0]).all()
