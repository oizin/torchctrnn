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
        raise Warning("ODENet's forward method only takes args: hidden,t and input. You also have {}".format(extra_args))
    # check for ['input', 't', 'hidden'] pattern in forward
    check_res = [chk for chk in check_args if chk not in args]
    if len(check_res) > 0:
        print("ODENet's forward method missing args: {}. These are assumed not applicable".format(check_res))

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

# class NeuralODE(nn.Module):
#     def __init__(self,vector_field:nn.Module,time_dependent=False,data_dependent=False,backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options):
#         if type(vector_field) == nn.Sequential:
#             _NeuralODEfromSequential(vector_field,time_dependent,data_dependent,backend,solver,atol, rtol,solver_options)
#         elif type(vector_field) == nn.Module:
#             _NeuralODEfromModule(vector_field,backend,solver,atol, rtol,solver_options)

# class _NeuralODEfromModule(nn.Module):
#     def __init__(self,vector_field:nn.Module,backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options):
#         """
#         TODO
#         ...
#         A feedforward network
#         """
#         super(_NeuralODEfromModule,self).__init__()
#         self.vector_field = vector_field
#         self.solver = solver
#         self.solver_options=solver_options
#         self.backend=backend
#         self.has_t_arg=False
#         self.has_input_arg=False
#         self.sequential = False
#         self.atol = atol
#         self.rtol = rtol

#         # check what arguments func (the vector field) takes:
#         args = inspect.getfullargspec(vector_field.forward)[0]
#         check_args = ['self','input', 't', 'hidden']
#         assert 'hidden' in args
#         # check for ['input', 't', 'hidden'] pattern in forward
#         extra_args = [a for a in args if a not in check_args]
#         if len(extra_args) > 0:
#             raise Warning("ODENet's forward method only takes args: hidden,t and input. You also have {}".format(extra_args))
#         # check for ['input', 't', 'hidden'] pattern in forward
#         check_res = [chk for chk in check_args if chk not in args]
#         if len(check_res) > 0:
#             print("ODENet's forward method missing args: {}. These are assumed not applicable".format(check_res))
#         if 'input' in args:
#             self.has_input_arg = True
#         if 't' in args:
#             self.has_t_arg = True

#     def forward(self,*args,**kwargs):
#         output = self.vector_field.forward(*args,**kwargs)
#         return output


# class _NeuralODEfromSequential(nn.Module):
#     def __init__(self,vector_field:nn.Sequential,time_dependent=False,data_dependent=False,backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options):
#         """
#         TODO
#         ...
#         A feedforward network
#         """
#         super(_NeuralODEfromSequential,self).__init__()
#         self.vector_field = vector_field
#         self.solver = solver
#         self.solver_options=solver_options
#         self.backend=backend
#         self.has_t_arg = time_dependent
#         self.has_input_arg = data_dependent
#         self.atol = atol
#         self.rtol = rtol

#         # infer hidden size
#         list(vector_field.parameters())

#     def forward(self,*args,**kwargs):
#         z = torch.cat(args + tuple(kwargs.values()),1) 
#         output = self.vector_field.forward(z)  
#         return output