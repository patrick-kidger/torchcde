import signatory
import torch
import torchcde


def test_with_linear_interpolation():
    for depth in (1, 2, 3, 4):
        compute_logsignature = signatory.Logsignature(depth)
        for pieces in (1, 2, 3, 5, 10):
            num_channels = torch.randint(low=1, high=4, size=(1,)).item()
            start = torch.rand(1).item() * 10 - 5
            end = torch.rand(1).item() * 10 - 5
            start, end = min(start, end), max(start, end)
            x_ = [torch.randn(1, num_channels, dtype=torch.float64)]
            logsignatures = []
            for _ in range(pieces):
                x = torch.randn(4, num_channels, dtype=torch.float64)
                logsignature = compute_logsignature(torch.cat([x_[-1][-1:], x]).unsqueeze(0))
                x_.append(x)
                logsignatures.append(logsignature)

            t = torch.linspace(start, end, 1 + 4 * pieces, dtype=torch.float64)
            x = torch.cat(x_)

            window_length = (end - start) / pieces
            logsig_t, logsig_x = torchcde.logsignature_windows(t, x, depth, window_length)
            coeffs = torchcde.linear_interpolation_coeffs(logsig_t, logsig_x)
            X = torchcde.LinearInterpolation(logsig_t, coeffs)

            point = 1
            for logsignature in logsignatures:
                interp_logsignature = X.derivative(t[point])
                assert interp_logsignature.allclose(logsignature)
                point += 4
