import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable,Union
from torch.nn.utils.parametrizations import spectral_norm
from . import utils
import inspect

def _check_lipschitz(net):
    return None

def _check_initial_condition(net):
    return None

class NeuralFlow(nn.Module):

    def __init__(self,flow:Union[nn.Module,nn.Sequential],time_func='none',times_dependent=None,data_dependent=None):
        """
        Arbitrary neural flow architecture
        
        """
        super(NeuralFlow,self).__init__()
        if type(flow) == nn.Sequential:
            assert type(times_dependent) == bool
            assert type(data_dependent) == bool
            self.sequential = True
        else:
            self.sequential = False
            args = inspect.getfullargspec(flow.forward)[0]
            utils._inspect_nn_args(args)
            if 'input' in args:
                data_dependent = True 
            if 'times' in args:
                times_dependent = True
        
        self.flow = flow
        self.has_times_arg=times_dependent
        self.has_input_arg=data_dependent

    def forward(self,*args,**kwargs):
        if self.sequential:
            z = torch.cat(args + tuple(kwargs.values()),1) 
            output = self.flow.forward(z)
        else:
            output = self.flow.forward(*args,**kwargs)
        return output

#     def forward(self,*args,**kwargs):
#         if self.sequential:
#             z = torch.cat(args + tuple(kwargs.values()),1) 
#             output = self.vector_field.forward(z)
#         else:
#             output = self.vector_field.forward(*args,**kwargs)
#         return output

# class ResNetFlow(nn.Module):
#     def __init__(self,input_size,hidden_size,n_power_iterations=1):
#         super(ResNetFlow,self).__init__()
#         self.n_power_iterations = n_power_iterations
#         self.net = nn.Sequential(
#                         spectral_norm(nn.Linear(input_size+1,hidden_size),n_power_iterations=self.n_power_iterations),
#                         nn.Tanh(),
#                         spectral_norm(nn.Linear(hidden_size,input_size),n_power_iterations=self.n_power_iterations),
#                         nn.Tanh()
#         )
#         self.time_func = nn.Tanh()

#     def forward(self,hidden,times,dt,input_ode):
#         return hidden + self.time_func(dt)*self.net(torch.cat((hidden,dt),1))

# class iResNetBlock(nn.Module):
#     def __init__(self,num_layers,hidden_size,final_activation='none',activation='tanh',n_power_iterations=1):
#         self.activation = activation_func(activation)
#         self.final_activation = activation_func(final_activation)
#         self.block = nn.ModuleList()
#         self.n_power_iterations = n_power_iterations

#         for layer in range(num_layers):
#             if layer < (num_layers - 1):
#                 self.block.append(
#                     nn.Sequential(
#                         spectral_norm(nn.Linear(hidden_size,hidden_size),self.n_power_iterations),
#                         activation_func(self.activation),
#                 ))
#             elif layer == (num_layers - 1):
#                 self.block.append(
#                     nn.Sequential(
#                         spectral_norm(nn.Linear(hidden_size,hidden_size),self.n_power_iterations),
#                         activation_func(self.final_activation),
#                 ))


#     def forward(self,z,t):
#         z + self.block(z,t)

# class iResNet(nn.Module):
#     def __init(self):
#         pass

#     def forward(self):
#         pass


# class GRUFlow(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self):
#         pass
