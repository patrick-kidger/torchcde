# torchcontroldiffeq
This library provides differentiable solvers for integrating controlled differential equations. Everything is implemented in PyTorch so there is full GPU support. The solvers are capable of utilising adjoint backpropagation for improved memory efficiency.

For example, this allows for extending Neural ODE models to time series.

_Powered by the excellent [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library._

## Installation

```bash
pip install git+https://github.com/patrick-kidger/torchcontroldiffeq.git
```

## Example
We encourage those interested in using this library to look at [example.py](./example/example.py), which demonstrates how to train a Neural CDE model to predict the chirality of a spiral.

## Documentation

TODO

## Citation
If you use this library in your work, we would appreciate a citation:

```bibtex
@article{kidger2020neuralcde,
    author={Kidger, Patrick and Morrill, James and Foster, James and Lyons, Terry},
    title={{Neural Controlled Differential Equations for Irregular Time Series}},
    year={2020},
    journal={arXiv:2005.08926}
}
```

## Extending the library
If you're interested in extending `torchcontroldiffeq` then have a look at [EXTENDING.md](./EXTENDING.md) for extra help on how to do this.

## Differences to `controldiffeq`
If you've used the [`controldiffeq`](https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq) library before then a couple things have been changed. See [DIFFERENCES.md](./DIFFERENCES.md).