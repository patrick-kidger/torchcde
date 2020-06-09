import functools as ft
import pytest
import torch
import torchcontroldiffeq


def test_computed_parameter():
    x = torch.rand(1, requires_grad=True).clone()
    y = torch.rand(1, requires_grad=True).clone()

    class TestPath(torchcontroldiffeq.Path):
        def __init__(self):
            with pytest.raises(ValueError):
                # ComputedParameter isn't appropriate here; should error
                self.should_fail = torchcontroldiffeq.ComputedParameter(torch.rand(1, requires_grad=True))

            with pytest.raises(RuntimeError):
                # Before super().__init__(); should error
                self.should_fail = torchcontroldiffeq.ComputedParameter(torch.rand(1, requires_grad=True).clone())

            # These things only exist after super().__init__(); should error
            assert not hasattr(self, '_controldiffeq_finalised')
            assert not hasattr(self, '_controldiffeq_computed_parameters')

            super(TestPath, self).__init__()

            # Should be False whilst we're still in __init__
            assert not self._controldiffeq_finalised

            # Can assign ComputedParameters
            self.variable = torchcontroldiffeq.ComputedParameter(x)
            assert self.variable is x
            assert set(self._controldiffeq_computed_parameters.keys()) == {'variable'}
            assert self._controldiffeq_computed_parameters['variable'] is x

            # Can reassign ComputedParameters
            self.variable = torchcontroldiffeq.ComputedParameter(y)
            assert self.variable is y
            assert set(self._controldiffeq_computed_parameters.keys()) == {'variable'}
            assert self._controldiffeq_computed_parameters['variable'] is y

            # Can delete ComputedParameters
            del self.variable
            assert len(self._controldiffeq_computed_parameters) == 0

            # Can reassign non-ComputedParameters over ComputedParameters
            self.variable = torchcontroldiffeq.ComputedParameter(x)
            assert self.variable is x
            assert set(self._controldiffeq_computed_parameters.keys()) == {'variable'}
            assert self._controldiffeq_computed_parameters['variable'] is x
            self.variable = None
            assert self.variable is None
            assert len(self._controldiffeq_computed_parameters) == 0

            # Can reassign ComputedParameters over non-ComputedParameters
            self.variable = torchcontroldiffeq.ComputedParameter(x)
            assert self.variable is x
            assert set(self._controldiffeq_computed_parameters.keys()) == {'variable'}
            assert self._controldiffeq_computed_parameters['variable'] is x

        def derivative(self, t):
            pass

    try:
        test_path = TestPath()
    except Exception:
        pytest.fail()
    # That the reassignment of ComputedParameters occurs successfully
    assert set(test_path._controldiffeq_computed_parameters.keys()) == {'variable'}
    assert test_path._controldiffeq_computed_parameters['variable'] is not x

    # Should be True after __init__
    assert test_path._controldiffeq_finalised

    # Can't assign ComputedParameters
    with pytest.raises(RuntimeError):
        test_path.variable2 = torchcontroldiffeq.ComputedParameter(x)
    assert set(test_path._controldiffeq_computed_parameters.keys()) == {'variable'}

    # Can't reassign ComputedParameters
    with pytest.raises(RuntimeError):
        test_path.variable = torchcontroldiffeq.ComputedParameter(y)
    assert set(test_path._controldiffeq_computed_parameters.keys()) == {'variable'}

    # Can delete ComputedParameters
    del test_path.variable
    assert len(test_path._controldiffeq_computed_parameters) == 0

    # Can't assign ComputedParameters where there's been a deletion
    with pytest.raises(RuntimeError):
        test_path.variable = torchcontroldiffeq.ComputedParameter(y)
    assert len(test_path._controldiffeq_computed_parameters) == 0

    test_path = TestPath()

    # Can reassign non-ComputedParameters over ComputedParameters
    test_path.variable = None
    assert test_path.variable is None
    assert len(test_path._controldiffeq_computed_parameters) == 0

    # Can't reassign ComputedParameters over non-ComputedParameters
    with pytest.raises(RuntimeError):
        test_path.variable = torchcontroldiffeq.ComputedParameter(x)
    assert len(test_path._controldiffeq_computed_parameters) == 0


def test_computed_parameter_graph():
    class TestPath(torchcontroldiffeq.Path):
        def __init__(self):
            super(TestPath, self).__init__()
            x = torch.rand(3, requires_grad=True)
            self.variable = torchcontroldiffeq.ComputedParameter(x.clone())
            self.variable2 = torchcontroldiffeq.ComputedParameter(self.variable.clone())

        def derivative(self, t):
            pass

    test_path = TestPath()
    grad = torch.autograd.grad(test_path.variable2.sum(), test_path.variable, allow_unused=True)
    grad2 = torch.autograd.grad(test_path.variable.sum(), test_path.variable2, allow_unused=True)
    # Despite one having been created from the other in __init__, they should have had views taken of them afterwards
    # to ensure that they're not in each other's computation graphs
    assert grad[0] is None
    assert grad2[0] is None


class _Func(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(_Func, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.variable = torch.nn.Parameter(torch.rand(1, 1, input_size))

    def forward(self, z):
        assert z.shape == (1, self.hidden_size)
        out = z.sigmoid().unsqueeze(-1) + self.variable
        assert out.shape == (1, self.hidden_size, self.input_size)
        return out


# Test that gradients can propagate through the controlling path at all
def test_grad_paths():
    for adjoint in (True, False):
        t = torch.linspace(0, 9, 10, requires_grad=True)
        path = torch.rand(1, 10, 3, requires_grad=True)
        coeffs = torchcontroldiffeq.natural_cubic_spline_coeffs(t, path)
        cubic_spline = torchcontroldiffeq.NaturalCubicSpline(coeffs)
        z0 = torch.rand(1, 3, requires_grad=True)
        func = _Func(input_size=3, hidden_size=3)
        t_ = torch.tensor([0., 9.], requires_grad=True)

        z = torchcontroldiffeq.cdeint(X=cubic_spline, func=func, z0=z0, t=t_, adjoint=adjoint, method='rk4')
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


# TODO: fix test
# Test that gradients flow back through multiple CDEs stacked on top of one another
def test_stacked_paths():
    log_ode_coeffs = ft.partial(torchcontroldiffeq.log_ode_coeffs, depth=3, window_length=5)
    coeff_paths = [(torchcontroldiffeq.linear_interpolation_coeffs, torchcontroldiffeq.LinearInterpolation),
                   (torchcontroldiffeq.natural_cubic_spline_coeffs, torchcontroldiffeq.NaturalCubicSpline),
                   (log_ode_coeffs, torchcontroldiffeq.LogODE)]
    for adjoint in (True, False):
        for first_coeffs, First in coeff_paths:
            for second_coeffs, Second in coeff_paths:
                for third_coeffs, Third in coeff_paths:
                    print(First.__name__, Second.__name__, Third.__name__)

                    first_t = torch.linspace(0, 999, 1000)
                    first_path = torch.rand(1, 1000, 4, requires_grad=True)
                    first_coeff = first_coeffs(first_t, first_path)
                    first_X = First(first_coeff)
                    first_func = _Func(input_size=30 if first_coeffs is log_ode_coeffs else 4, hidden_size=4)
                    print('a')

                    second_t = torch.linspace(0, 999, 100)
                    second_path = torchcontroldiffeq.cdeint(X=first_X, func=first_func, z0=torch.rand(1, 4), t=second_t,
                                                            adjoint=adjoint, method='rk4')
                    second_coeff = second_coeffs(second_t, second_path)
                    second_X = Second(second_coeff)
                    second_func = _Func(input_size=30 if second_coeffs is log_ode_coeffs else 4, hidden_size=4)
                    print('b')

                    third_t = torch.linspace(0, 999, 10)
                    third_path = torchcontroldiffeq.cdeint(X=second_X, func=second_func, z0=torch.rand(1, 4), t=third_t,
                                                           adjoint=adjoint, method='rk4')
                    third_coeff = third_coeffs(third_t, third_path)
                    third_X = Third(third_coeff)
                    third_func = _Func(input_size=30 if third_coeffs is log_ode_coeffs else 4, hidden_size=5)
                    print('c')

                    fourth_t = torch.tensor([0, 999.])
                    fourth_path = torchcontroldiffeq.cdeint(X=third_X, func=third_func, z0=torch.rand(1, 5), t=fourth_t,
                                                            adjoint=adjoint, method='rk4')
                    print('d')
                    assert first_func.variable.grad is None
                    assert second_func.variable.grad is None
                    assert third_func.variable.grad is None
                    assert first_path.grad is None
                    print('e')
                    fourth_path[:, -1].sum().backward()
                    print('f')
                    assert isinstance(first_func.variable.grad, torch.Tensor)
                    assert isinstance(second_func.variable.grad, torch.Tensor)
                    assert isinstance(third_func.variable.grad, torch.Tensor)
                    assert isinstance(first_path.grad, torch.Tensor)


# Tests that the trick in which we use detaches in the backward pass if possible, does in fact work
def test_detach_trick():
    t = torch.linspace(0, 9, 10)
    path = torch.rand(1, 10, 3)
    coeffs = torchcontroldiffeq.natural_cubic_spline_coeffs(t, path)
    cubic_spline = torchcontroldiffeq.NaturalCubicSpline(coeffs)
    func = _Func(input_size=3, hidden_size=3)

    for adjoint in (True, False):
        variable_grads = []
        z0 = torch.rand(1, 3)
        for t_grad in (True, False):
            t_ = torch.tensor([0., 9.], requires_grad=t_grad)
            z = torchcontroldiffeq.cdeint(X=cubic_spline, z0=z0, func=func, t=t_, adjoint=adjoint, method='rk4')
            z[:, -1].sum().backward()
            variable_grads.append(func.variable.grad.clone())
            func.variable.grad.zero_()

        for elem in variable_grads[1:]:
            assert (elem == variable_grads[0]).all()
