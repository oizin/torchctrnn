# torchctrnn
## _Continuous time RNNs in pytorch_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

torchctrnn is a lightweight package for using neural ODE based continuous time RNNs and related methods with pytorch and torchdiffeq

## Features

- Use `torchctrnn.ODERNNCell` like you would `torch.RNNCell`
- Unified framework

forward(self,input_update,h_0,times,input_ode=None,n_intermediate=0):   

## Installation

torchctrnn requires pytorch and torchdiffeq to run

Install the dependencies and devDependencies and start the server.

```sh
pip install torchctrnn
```

## Development

Want to contribute? Great!

## License

MIT