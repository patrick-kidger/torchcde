import math
import numpy as np
import torch


def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)


def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)


def validate_input_path(x, t):
    if not x.is_floating_point():
        raise ValueError("X must both be floating point.")

    if x.ndimension() < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels. It instead has "
                         "shape {}.".format(tuple(x.shape)))

    if t is None:
        t = torch.linspace(0, x.size(-2) - 1, x.size(-2), dtype=x.dtype, device=x.device)

    if not t.is_floating_point():
        raise ValueError("t must both be floating point.")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional. It instead has shape {}.".format(tuple(t.shape)))
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")
        prev_t_i = t_i

    if x.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t. X has shape {} and t has shape {}, "
                         "corresponding to time dimensions of {} and {} respectively."
                         .format(tuple(x.shape), tuple(t.shape), x.size(-2), t.size(0)))

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2. It instead has shape {}, corresponding to a "
                         "time dimension of size {}.".format(tuple(t.shape), t.size(0)))

    return t


def forward_fill(x, fill_index=-2):
    """Forward fills data in a torch tensor of shape (..., length, input_channels) along the length dim.

    Arguments:
        x: tensor of values with first channel index being time, of shape (..., length, input_channels), where ... is
            some number of batch dimensions.
        fill_index: int that denotes the index to fill down. Default is -2 as we tend to use the convention (...,
            length, input_channels) filling down the length dimension.

    Returns:
        A tensor with forward filled data.
    """
    # Checks
    assert isinstance(x, torch.Tensor)
    assert x.dim() >= 2

    mask = torch.isnan(x)
    if mask.any():
        cumsum_mask = (~mask).cumsum(dim=fill_index)
        cumsum_mask[mask] = 0
        _, index = cumsum_mask.cummax(dim=fill_index)
        x = x.gather(dim=fill_index, index=index)

    return x


class TupleControl(torch.nn.Module):
    def __init__(self, *controls):
        super(TupleControl, self).__init__()

        if len(controls) == 0:
            raise ValueError("Expected one or more controls to batch together.")

        self._interval = controls[0].interval
        grid_points = controls[0].grid_points
        same_grid_points = True
        for control in controls[1:]:
            if (control.interval != self._interval).any():
                raise ValueError("Can only batch togehter controls over the same interval.")
            if same_grid_points and (control.grid_points != grid_points).any():
                same_grid_points = False

        if same_grid_points:
            self._grid_points = grid_points
        else:
            self._grid_points = None

        self.controls = torch.nn.ModuleList(controls)

    @property
    def interval(self):
        return self._interval

    @property
    def grid_points(self):
        if self._grid_points is None:
            raise RuntimeError("Batch of controls have different grid points.")
        return self._grid_points

    def evaluate(self, t):
        return tuple(control.evaluate(t) for control in self.controls)

    def derivative(self, t):
        return tuple(control.derivative(t) for control in self.controls)
