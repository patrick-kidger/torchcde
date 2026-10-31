import pytest
import torch
import torchcde.misc  # testing an implementation detail


def test_cheap_stack():
    for num in range(1, 4):
        for dim in (-2, -1, 0, 1):
            xs = [torch.rand(1, 1) for _ in range(num)]
            s = torchcde.misc.cheap_stack(xs, dim)
            s2 = torch.stack(xs, dim)
            assert s.shape == s2.shape
            assert (s == s2).all()


def test_tridiagonal_solve():
    for _ in range(5):
        size = torch.randint(low=2, high=10, size=(1,)).item()
        diagonal = torch.randn(size, dtype=torch.float64)
        upper = torch.randn(size - 1, dtype=torch.float64)
        lower = torch.randn(size - 1, dtype=torch.float64)
        A = torch.zeros(size, size, dtype=torch.float64)
        A[range(size), range(size)] = diagonal
        A[range(1, size), range(size - 1)] = lower
        A[range(size - 1), range(1, size)] = upper
        b = torch.randn(size, dtype=torch.float64)
        x = torchcde.misc.tridiagonal_solve(b, upper, diagonal, lower)
        mul = A @ x
        assert mul.allclose(b)


def test_forward_fill():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        # Check ffill
        for N, L, C in [(1, 5, 3), (2, 2, 2), (3, 2, 1)]:
            x = torch.randn(N, L, C).to(device)
            # Drop mask
            tensor_num = x.numel()
            mask = torch.randperm(tensor_num)[:int(0.3 * tensor_num)].to(device)
            x.view(-1)[mask] = float('nan')
            x_ffilled = x.clone().float()
            for i in range(0, x.size(0)):
                for j in range(x.size(1)):
                    for k in range(x.size(2)):
                        non_nan = x_ffilled[i, :j + 1, k][~torch.isnan(x[i, :j + 1, k])]
                        input_val = non_nan[-1].item() if len(non_nan) > 0 else float('nan')
                        x_ffilled[i, j, k] = input_val
            x_ffilled_actual = torchcde.misc.forward_fill(x)
            assert x_ffilled.allclose(x_ffilled_actual, equal_nan=True)


def test_tuple_control_different_number_of_grid_points():
    t_a = torch.linspace(0, 1, 5)
    t_b = torch.linspace(0, 1, 3)
    X_a = torchcde.LinearInterpolation(torchcde.linear_interpolation_coeffs(torch.rand(5, 2), t_a), t_a)
    X_b = torchcde.LinearInterpolation(torchcde.linear_interpolation_coeffs(torch.rand(3, 1), t_b), t_b)
    X = torchcde.TupleControl(X_a, X_b)
    assert torch.equal(X.interval, X_a.interval)
    with pytest.raises(RuntimeError, match="different grid points"):
        X.grid_points

    def func(t, z):
        z_a, z_b = z
        return z_a.tanh().unsqueeze(-1).expand(3, 2), z_b.tanh().unsqueeze(-1)

    z0 = torch.rand(3), torch.rand(4)
    z_a, z_b = torchcde.cdeint(X=X, func=func, z0=z0, t=X.interval, adjoint=False, method="rk4",
                               options=dict(step_size=0.1))
    assert z_a.shape == (2, 3)
    assert z_b.shape == (2, 4)
