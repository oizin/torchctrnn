import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
import inspect

# class _ODENetBase(nn.Module):
#     """
#     purpose of this class?
#     """
#     def __init__(self,ODENet):
#         super(_ODENetBase, self).__init__()
#         self.has_input_arg = False
#         self.has_t_arg = False

#         # check what arguments ODENet (the vector field) takes:
#         args = inspect.getfullargspec(ODENet.forward)[0]
#         check_args = ['input', 't', 'hidden']
#         check_res = [chk for chk in check_args if chk not in args]
#         if len(check_res) > 0:
#             raise Warning("ODENet's forward method missing args: {}. These are assumed not applicable".format(check_res))
#         if 'input' in args:
#             self.has_input_arg = True
#         if 't' in args:
#             self.has_t_arg = True
#         self.net = ODENet


class ODENet(nn.Module):
    def __init__(self,vector_field:nn.Module,backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options):
        """
        TODO
        ...
        A feedforward network
        """
        super(ODENet,self).__init__()
        self.vector_field = vector_field
        self.solver = solver
        self.solver_options=solver_options
        self.backend=backend
        self.has_t_arg=False
        self.has_input_arg=False
        self.sequential = False
        self.atol = atol
        self.rtol = rtol

        # check what arguments func (the vector field) takes:
        args = inspect.getfullargspec(vector_field.forward)[0]
        check_args = ['input', 't', 'hidden']
        assert 'hidden' in args
        # check for ['input', 't', 'hidden'] pattern in forward
        extra_args = [a for a in args if a not in check_args]
        if len(extra_args) > 0:
            raise Warning("ODENet's forward method only takes args: hidden,t and input. You have {}".format(extra_args))
        check_res = [chk for chk in check_args if chk not in args]
        # check for ['input', 't', 'hidden'] pattern in forward
        if len(check_res) > 0:
            raise Warning("ODENet's forward method missing args: {}. These are assumed not applicable".format(check_res))
        if 'input' in args:
            self.has_input_arg = True
        if 't' in args:
            self.has_t_arg = True

    def forward(self,**inputs):
        output = self.vector_field.forward(*inputs)
        return output


class ODENetfromSequential(nn.Module):
    def __init__(self,vector_field:nn.Sequential,time_dependent=False,data_dependent=False,backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options):
        """
        TODO
        ...
        A feedforward network
        """
        super(ODENetfromSequential,self).__init__()
        self.vector_field = vector_field
        self.solver = solver
        self.solver_options=solver_options
        self.backend=backend
        self.has_t_arg = time_dependent
        self.has_input_arg = data_dependent
        self.atol = atol
        self.rtol = rtol

        # infer hidden size
        list(vector_field.parameters())

    def forward(self,**inputs):
        z = torch.cat(tuple(inputs.values()),1) 
        output = self.vector_field.forward(z)  
        return output