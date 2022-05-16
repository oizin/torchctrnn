# torchctrnn


<!-- badges: start -->

[![Test results](https://github.com/oizin/torchctrnn/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/oizin/torchctrnn/actions/workflows/run_tests.yaml)
[![codecov](https://codecov.io/github/oizin/torchctrnn/branch/main/graphs/badge.svg)](https://codecov.io/github/oizin/torchctrnn)
[![Documentation Status](https://readthedocs.org/projects/torchctrnn/badge/?version=latest)](https://torchctrnn.readthedocs.io/en/latest/?badge=latest)
<!-- badges: end -->

## Continuous time RNNs in PyTorch

torchctrnn is a PyTorch library dedicated to continuous time recurrent neural networks.

## Installation

To install latest on GitHub:

```
pip install git+https://github.com/oizin/torchctrnn
```

## Basic usage

- Use `torchctrnn.ODERNNCell` similar to how you would use `torch.RNNCell` with the 
addition of specifying the neural ODE

```python
forward(self,input_update,h_0,times,input_ode=None,n_intermediate=0):   
```

## Documentation

https://torchctrnn.readthedocs.io/en/latest/

## Testing

```python
# library
pytest
# tutorials
pytest --nbmake ./tutorials
```

## Development

Want to contribute? Get in touch.

## License

MIT