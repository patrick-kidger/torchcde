# torchcontroldiffeq
This library provides differentiable solvers for integrating controlled differential equations. Everything is implemented in PyTorch so there is full GPU support. The solvers utilise adjoint backpropagation for improved memory efficiency.

For example, this allows for extending Neural ODE models to time series.

_Powered by the excellent [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library._

## Installation

TODO

## Example
We encourage those interested in using this library to look at [example.py](https://github.com/patrick-kidger/torchcontroldiffeq/blob/master/torchcontroldiffeq/example.py), which demonstrates how to train a Neural CDE model to predict the chirality of a spiral.

## Documentation

TODO

## Citation
If you use this library in your research, we would appreciate a citation:

TODO

## Extending the library
If you're interested in extending `torchcontroldiffeq` then have a look at [EXTENDING.md](https://github.com/patrick-kidger/torchcontroldiffeq/blob/master/EXTENDING.md) for extra help on how to do this.