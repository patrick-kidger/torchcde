import torch
import torchcontroldiffeq.misc  # testing an implementation detail


def test_identity():
    x = torch.rand(1)
    y = torchcontroldiffeq.misc.identity(x)
    assert x.shape == y.shape
    assert (x == y).all()
    assert y is not x


def test_cheap_stack():
    for num in range(1, 4):
        for dim in (-2, -1, 0, 1):
            xs = [torch.rand(1, 1) for _ in range(num)]
            s = torchcontroldiffeq.misc.cheap_stack(xs, dim)
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
        x = torchcontroldiffeq.misc.tridiagonal_solve(b, upper, diagonal, lower)
        mul = A @ x
        assert mul.allclose(b)
