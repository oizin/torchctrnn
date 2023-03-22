import torch
import torch.nn as nn
from torch import Tensor
from . import utils
from typing import Callable,Union
import inspect

class NeuralODE(nn.Module):

    def __init__(self,vector_field:Union[nn.Module,nn.Sequential],time_func='none',delta_t_func='none',time_dependent=None,data_dependent=None,backend='torchdiffeq',solver='dopri5',atol:float=1e-3, rtol:float=1e-3,solver_options={}):
        super(NeuralODE,self).__init__()
        # check arguments
        if type(vector_field) == nn.Sequential:
            assert type(time_dependent) == bool
            assert type(data_dependent) == bool
            self.sequential = True
        else:
            self.sequential = False
            args = inspect.getfullargspec(vector_field.forward)[0]
            utils._inspect_node_args(args)
            if 'input' in args:
                data_dependent = True 
            if 't' in args:
                time_dependent = True
        # properties
        self.vector_field = vector_field
        self.solver = solver
        self.solver_options=solver_options
        self.backend=backend
        self.has_t_arg=time_dependent
        self.has_input_arg=data_dependent
        self.atol = atol
        self.rtol = rtol
        if isinstance(time_func,str):
            self.time_func = utils.time_func(time_func)
        elif callable(time_func):
            self.time_func = time_func
        if isinstance(delta_t_func,str):
            self.delta_t_func = utils.time_func(delta_t_func)
        elif callable(delta_t_func):
            self.delta_t_func = delta_t_func

    def forward(self,*args,**kwargs):
        if self.sequential:
            z = torch.cat(args + tuple(kwargs.values()),1) 
            output = self.vector_field.forward(z)
        else:
            output = self.vector_field.forward(*args,**kwargs)
        return output