import torch
import torchcde

import markers


@markers.uses_signatory
def test_with_linear_interpolation():
    import signatory
    window_length = 4
    for depth in (1, 2, 3, 4):
        compute_logsignature = signatory.Logsignature(depth)
        for pieces in (1, 2, 3, 5, 10):
            num_channels = torch.randint(low=1, high=4, size=(1,)).item()
            x_ = [torch.randn(1, num_channels, dtype=torch.float64)]
            logsignatures = []
            for _ in range(pieces):
                x = torch.randn(window_length, num_channels, dtype=torch.float64)
                logsignature = compute_logsignature(torch.cat([x_[-1][-1:], x]).unsqueeze(0))
                x_.append(x)
                logsignatures.append(logsignature)

            x = torch.cat(x_)

            logsig_x = torchcde.logsig_windows(x, depth, window_length)
            coeffs = torchcde.linear_interpolation_coeffs(logsig_x)
            X = torchcde.LinearInterpolation(coeffs)

            point = 0.5
            for logsignature in logsignatures:
                interp_logsignature = X.derivative(point)
                assert interp_logsignature.allclose(logsignature)
                point += 1
