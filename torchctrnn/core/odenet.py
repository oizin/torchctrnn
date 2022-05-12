import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable,Union
import inspect

def _inspect_nn_args(args):
    check_args = ['self','input', 't', 'hidden']
    assert 'hidden' in args
    # check for ['input', 't', 'hidden'] pattern in forward
    extra_args = [a for a in args if a not in check_args]
    if len(extra_args) > 0:
        raise Warning("NeuralODE's forward method only takes args: hidden,t and input. You also have {}".format(extra_args))
    # check for ['input', 't', 'hidden'] pattern in forward
    check_res = [chk for chk in check_args if chk not in args]
    if len(check_res) > 0:
        print("NeuralODE's forward method missing args: {}. These are assumed not applicable".format(check_res))

class NeuralODE(nn.Module):

    def __init__(self,vector_field:Union[nn.Module,nn.Sequential],time_dependent=None,data_dependent=None,backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options):
        super(NeuralODE,self).__init__()
        # check arguments
        if type(vector_field) == nn.Sequential:
            assert type(time_dependent) == bool
            assert type(data_dependent) == bool
            self.sequential = True
        else:
            self.sequential = False
            args = inspect.getfullargspec(vector_field.forward)[0]
            _inspect_nn_args(args)
            if 'input' in args:
                self.has_input_arg = True
            if 't' in args:
                self.has_t_arg = True
        # properties
        self.vector_field = vector_field
        self.solver = solver
        self.solver_options=solver_options
        self.backend=backend
        self.has_t_arg=time_dependent
        self.has_input_arg=data_dependent
        self.atol = atol
        self.rtol = rtol

    def forward(self,*args,**kwargs):
        if self.sequential:
            z = torch.cat(args + tuple(kwargs.values()),1) 
            output = self.vector_field.forward(z)
        else:
            output = self.vector_field.forward(*args,**kwargs)
        return output