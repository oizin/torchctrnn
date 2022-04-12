# torchctrnn
# Under development
## _Continuous time RNNs in pytorch_

torchctrnn is a lightweight package for using neural ODE based continuous time RNNs and related methods with pytorch and torchdiffeq

## Features

- Use `torchctrnn.ODERNNCell` similar to how you would use `torch.RNNCell`

```python


forward(self,input_update,h_0,times,input_ode=None,n_intermediate=0):   
```

See the examples folder.

## Installation

torchctrnn requires pytorch and torchdiffeq to run

## Development

Want to contribute? Get in touch.

## License

MIT