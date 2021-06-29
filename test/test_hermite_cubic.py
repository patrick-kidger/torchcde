import torch
import torchcde


# Represents a random Hermite cubic spline
class _HermiteCubic:
    def __init__(self, batch_dims, num_channels, length=3):
        self.data = torch.randn(*batch_dims, length, num_channels, dtype=torch.float64)
        x_next = self.data[..., 1:, :]
        x_prev = self.data[..., :-1, :]
        delta_next = x_next - x_prev
        delta_prev = x_next[..., 1:, :] - x_prev[..., :-1, :]

        self.D = x_prev
        self.C = delta_prev
        self.B = 2 * (delta_next - delta_prev)
        self.A = -(delta_next - delta_prev)

    def _normalise_dims(self, t):
        a = self.a
        b = self.b
        c = self.c
        d1 = self.d1
        d2 = self.d2
        for _ in t.shape:
            a = a.unsqueeze(-2)
            b = b.unsqueeze(-2)
            c = c.unsqueeze(-2)
            d1 = d1.unsqueeze(-2)
            d2 = d2.unsqueeze(-2)
        t = t.unsqueeze(-1)
        d = torch.where(t >= 0, d2, d1)
        return a, b, c, d, t

    def evaluate(self):


if __name__ == '__main__':
