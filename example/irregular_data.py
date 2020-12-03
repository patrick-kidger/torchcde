######################
# Processing irregular data is usually rather tricky.
# One of the best bits about Neural CDEs is how it is instead really straightforward.
#
# Here we'll look at how you can handle:
# - variable-length sequences
# - irregular sampling
# - missing data
# - informative missingness
#
# In every case, the only thing that needs changing is the data preprocessing. You won't need to change your model at
# all.
######################

import torch
import torchcde


######################
# We begin with a helper function that solves an example CDE. This is going to be the final step of all of our examples.
######################
def _solve_cde(x):
    # x should be of shape (batch, length, channels)
    
    # Create dataset
    coeffs = torchcde.natural_cubic_spline_coeffs(x)

    # Create model
    batch_size = x.size(0)
    input_channels = x.size(2)
    hidden_channels = 4  # hyperparameter, we can pick whatever we want for this
    output_channels = 4  # e.g. to perform 4-way multiclass classification

    class F(torch.nn.Module):
        def __init__(self):
            super(F, self).__init__()
            self.linear = torch.nn.Linear(hidden_channels,
                                          hidden_channels * input_channels)

        def forward(self, t, z):
            return self.linear(z).view(batch_size, hidden_channels, input_channels)
        
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.initial = torch.nn.Linear(input_channels, hidden_channels)
            self.func = F()
            self.readout = torch.nn.Linear(hidden_channels, output_channels)
            
        def forward(self, coeffs):
            X = torchcde.NaturalCubicSpline(coeffs)
            X0 = X.evaluate(X.interval[0])
            z0 = self.initial(X0)
            zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
            zT = zt[:, -1]  # get the terminal value of the CDE
            return self.readout(zT)
        
    model = Model()
    
    # Run model
    model(coeffs)


def variable_length_data():
    ######################
    # Let's generate a batch of data without a consistent length.
    ######################

    def func(t):
        return torch.cos(t) + t ** 2  # Example function to get data from

    # Batch of three elements
    t1 = torch.linspace(0, 6, 7)
    t2 = torch.linspace(0, 9, 10)
    t3 = torch.linspace(0, 5, 6)
    x1 = func(t1)
    x2 = func(t2)
    x3 = func(t3)

    # As always, don't forget to append time as an extra channel.
    x1 = torch.stack([t1, x1], dim=1)  # shape (7, 2)
    x2 = torch.stack([t2, x2], dim=1)  # shape (10, 2)
    x3 = torch.stack([t3, x3], dim=1)  # shape (6, 2)

    ######################
    # Now pad out every shorter sequence by _filling the last value forward_.
    # That's emphasised because using fill-forward here is absolutely crucial.
    ######################

    max_length = max(x1.size(0), x2.size(0), x3.size(0))  # 10
    input_channels = x1.size(1)  # 2
    x1 = torch.cat([x1, x1[-1].unsqueeze(0).expand(max_length - x1.size(0), input_channels)], dim=0)  # shape (10, 2)
    x2 = torch.cat([x2, x2[-1].unsqueeze(0).expand(max_length - x2.size(0), input_channels)], dim=0)  # shape (10, 2)
    x3 = torch.cat([x3, x3[-1].unsqueeze(0).expand(max_length - x3.size(0), input_channels)], dim=0)  # shape (10, 2)

    ######################
    # Batch everything together
    ######################
    x = torch.stack([x1, x2, x3])  # shape (3, 10, 2)

    ######################
    # Solve a Neural CDE
    ######################

    zT = _solve_cde(x)

    ######################
    # What's going on here? We just extracted the _final_ value of the CDE, despite having applied padding to our
    # sequences. Shouldn't we have had to get some of the intermediate values as well, to get the final value for each
    # individual batch element?
    #
    # Not so!
    # This is a neat trick: Remember that (in differential equation form), a CDE is given by:
    # dz/dt(t) = f(t, z)dX/dt(t)
    # So when we chose to use fill-forward to pad in our data, then the data is _constant_ over the padding. That means
    # that its derivative, dX/dt, is zero. Once the data stops changing, then the hidden state will stop changing as
    # well.
    #
    # Importantly: we applied padding _after_ appending time. If we did it the other way around then the time channel
    # would still keep changing, and this wouldn't work.
    #
    # Note that technically speaking, a cubic spline interpolation, being smooth, will still have small perturbations in
    # dX/dt: it won't _quite_ be zero. Practically speaking this is unlikely to be an issue, but if you prefer then use
    # linear interpolation instead, which will set dX/dt to exactly zero.
    ######################


def irregular_sampling():
    ######################
    # Now let's generate some data that was sampled at different time points.
    ######################

    def func(t):
        return torch.cos(t) + t ** 2  # Example function to get data from

    # Batch of three elements. Each element is sampled at 10 different times.
    t1 = torch.rand(10).sort().values
    t2 = torch.rand(10).sort().values
    t3 = torch.rand(10).sort().values
    x1 = func(t1)
    x2 = func(t2)
    x3 = func(t3)

    # As always, don't forget to append time as an extra channel.
    x1 = torch.stack([t1, x1], dim=1)  # shape (10, 2)
    x2 = torch.stack([t2, x2], dim=1)  # shape (10, 2)
    x3 = torch.stack([t3, x3], dim=1)  # shape (10, 2)

    ######################
    # Batch everything together
    ######################
    x = torch.stack([x1, x2, x3])  # shape (3, 10, 2)

    ######################
    # Solve a Neural CDE
    ######################

    zT = _solve_cde(x)

    ######################
    # We don't have to care that things were sampled at different time points. Time is just another channel of the data.
    ######################


def missing_data():
    ######################
    # Now for an example with missing data. This is essentially a generalisation of the irregular sampling case to
    # multiple channels.
    ######################

    def func(t):
        return torch.cos(t) + t ** 2  # Example function to get data from

    # Batch of three elements, each of two channels. Each element and channel is sampled at 10 different times.
    t1a, t1b = torch.rand(10).sort().values, torch.rand(10).sort().values
    t2a, t2b = torch.rand(10).sort().values, torch.rand(10).sort().values
    t3a, t3b = torch.rand(10).sort().values, torch.rand(10).sort().values
    x1a, x1b = func(t1a), func(t1b)
    x2a, x2b = func(t2a), func(t2b)
    x3a, x3b = func(t3a), func(t3b)

    ######################
    # This looks a bit tricky. We've got a batch element x1, whose first channel was sampled at `t1a` and whose
    # second channel was sampled at `t1b`.
    # However as always, Neural CDEs make life simple.
    ######################

    # First get all the times that the first batch element was sampled at.
    t1, sort_indices1 = torch.cat([t1a, t1b]).sort()
    # Now add NaNs to each channel where the other channel was sampled.
    x1a_ = torch.cat([x1a, torch.full_like(x1b, float('nan'))])[sort_indices1]
    x1b_ = torch.cat([torch.full_like(x1a, float('nan')), x1b])[sort_indices1]
    # Stack them together. As always, don't forget to also append time as an extra channel.
    x1 = torch.stack([t1, x1a_, x1b_], dim=1)

    # Now do the same for the other batch elements.
    t2, sort_indices2 = torch.cat([t2a, t2b]).sort()
    x2a_ = torch.cat([x2a, torch.full_like(x2b, float('nan'))])[sort_indices2]
    x2b_ = torch.cat([torch.full_like(x2a, float('nan')), x2b])[sort_indices2]
    x2 = torch.stack([t2, x2a_, x2b_], dim=1)
    t3, sort_indices3 = torch.cat([t3a, t3b]).sort()
    x3a_ = torch.cat([x3a, torch.full_like(x3b, float('nan'))])[sort_indices3]
    x3b_ = torch.cat([torch.full_like(x3a, float('nan')), x3b])[sort_indices3]
    x3 = torch.stack([t3, x3a_, x3b_], dim=1)

    ######################
    # Batch everything together
    ######################
    x = torch.stack([x1, x2, x3])  # shape (3, 10, 3)

    ######################
    # Solve a Neural CDE
    ######################

    zT = _solve_cde(x)

    ######################
    # Let's recap what's happened here.
    # We indicated missing values by putting in some NaNs in `x`.
    # Then when `natural_cubic_spline_coeffs` is called inside `_solve_cde`, it just did the interpolation over the
    # missing values. Job done.
    ######################


def informative_missingness():
    ######################
    # You may (or may not) be familiar with the problem of informative missingness: the idea that the sampling rate of
    # irregular data itself carries information. For example, doctors are more likely to make measurements of patients
    # they believe to be ill.
    #
    # So let's push the previous example a little further, and include this too.
    # We begin by generating dadta just like we did before.
    ######################

    def func(t):
        return torch.cos(t) + t ** 2  # Example function to get data from

    # For brevity we'll only generate a single batch element this time.
    t1a, t1b = torch.rand(10).sort().values, torch.rand(10).sort().values
    x1a, x1b = func(t1a), func(t1b)

    ######################
    # Now we generate observational masks. We also fill in missing values with NaN just like before.
    ######################

    t1, sort_indices1 = torch.cat([t1a, t1b]).sort()
    mask1a = torch.cat([torch.ones_like(x1a), torch.zeros_like(x1b)])[sort_indices1]
    mask1b = torch.cat([torch.zeros_like(x1a), torch.ones_like(x1b)])[sort_indices1]
    x1a_ = torch.cat([x1a, torch.full_like(x1b, float('nan'))])[sort_indices1]
    x1b_ = torch.cat([torch.full_like(x1a, float('nan')), x1b])[sort_indices1]

    ######################
    # Now stack everything together as before, using the cumulative sum of the masks.
    ######################

    x1 = torch.stack([t1, x1a_, x1b_, mask1a.cumsum(dim=0), mask1b.cumsum(dim=0)], dim=1)

    ######################
    # And now do exactly as we did before: repeat the same procedure for every batch element, stack them together, solve
    # a CDE, and so on.
    #
    # So let's explain what's happening here.
    # The basic idea of appending observational masks is pretty standard when concerned about informative missingness:
    # we need to tell the network whether that data point was actually observed or not.
    # Here, the only difference is that we also take a cumulative sum. This produces a sequence that increases every
    # time we make an observation.
    # When we interpolate that channel into a path, we'll get a cumulative intensity for how often that channel was
    # observed.
    # As it is the derivative of the path (interpolating the data) that is used inside the Neural CDE, then this tells
    # us the sampling intensity itself.
    # In other words, the derivative and the cumulative sum "cancel out".
    ######################


######################
# If you've made it this far, and followed everything, then congratulations!
# All that remains is, of course, to leave one final exercise for the reader:
#
# Try putting all of these examples together, to produce something that can handle variable lengths, informative
# missingness and so on, all at the same time.
#
# Best of luck using Neural CDEs.
######################
