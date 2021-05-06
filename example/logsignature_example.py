######################
# In this script we code up a Neural CDE using the log-ode method for a long time series thus becoming a Neural RDE.
# This paper describing this methodology can be found at https://arxiv.org/pdf/2009.08295.pdf
# This method assumes familiarity with the standard Neural CDE example at `time_series_classification.py`. We will only
# describe the differences from that example.
######################
import time
import torch
import torchcde
from time_series_classification import NeuralCDE, get_data


def _train(train_X, train_y, test_X, test_y, depth, num_epochs, window_length):
    # Time the training process
    start_time = time.time()

    # Logsignature computation step
    train_logsig = torchcde.logsig_windows(train_X, depth, window_length=window_length)
    print("Logsignature shape: {}".format(train_logsig.size()))

    model = NeuralCDE(
        input_channels=train_logsig.size(-1), hidden_channels=8, output_channels=1, interpolation="linear"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    train_coeffs = torchcde.linear_interpolation_coeffs(train_logsig)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch: {}   Training loss: {}".format(epoch, loss.item()))

    # Remember to compute the logsignatures of the test data too!
    test_logsig = torchcde.logsig_windows(test_X, depth, window_length=window_length)
    test_coeffs = torchcde.linear_interpolation_coeffs(test_logsig)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print("Test Accuracy: {}".format(proportion_correct))

    # Total time
    elapsed = time.time() - start_time

    return proportion_correct, elapsed


def print_heading(message):
    # Print a message inbetween rows of #'s
    string_sep = "#" * 50
    print("\n" + string_sep + "\n{}\n".format(message) + string_sep)


def main(num_epochs=15):
    ######################
    # Here we load a high frequency version of the spiral data using in `torchcde.example`. Each sample contains 5000
    # time points. This is too long to sensibly expect a Neural CDE to handle, training time will be very long and it
    # will struggle to remember information from early on in the sequence.
    ######################
    num_timepoints = 5000
    train_X, train_y = get_data(num_timepoints=num_timepoints)
    test_X, test_y = get_data(num_timepoints=num_timepoints)

    ######################
    # We test the model over logsignature depths [1, 2, 3] with a window length of 50. This reduces the effective
    # length of the path to just 100. The only change is an application of `torchcde.logsig_windows`

    # The raw signal has 3 input channels. Taking logsignatures of depths [1, 2, 3] results in a path of logsignatures
    # of dimension [3, 6, 14] respectively. We see that higher logsignature depths contain more information about the
    # path over the intervals, at a cost of increased numbers of channels.
    ######################
    depths = [1, 2, 3]
    window_length = 50
    accuracies = []
    training_times = []
    for depth in depths:
        print_heading('Running for logsignature depth: {}'.format(depth))
        acc, elapsed = _train(
            train_X, train_y, test_X, test_y, depth, num_epochs, window_length
        )
        training_times.append(elapsed)
        accuracies.append(acc)

    # Finally log the results to the console for a comparison
    print_heading("Final results")
    for acc, elapsed, depth in zip(accuracies, training_times, depths):
        print(
            "Depth: {}\n\tAccuracy on test set: {:.1f}%\n\tTime per epoch: {:.1f}s".format(
                depth, acc * 100, elapsed / num_epochs
            )
        )


if __name__ == "__main__":
    main()
