import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from torch.nn.utils.parametrizations import spectral_norm
from .utils import activation_func

def _check_lipschitz(net):
    return None

def _check_initial_condition(net):
    return None

class NeuralFlow(nn.Module):

    def __init__(self,vector_field:nn.Module):
        """
        
        
        """
        self.vector_field = vector_field
        _check_lipschitz(vector_field)
        _check_initial_condition(vector_field)


    def forward(self):
        pass


# class ResNetFlow(nn.Module):
#     def __init__(self,flow_layers:int=1,hidden_layers:int=1):
#         super(ResNetFlow, self).__init__()
#         self.vector_field = nn.ModuleList()
#         for flayer in range(flow_layers):
#             for hlayer in range(hidden_layers):
#                 self.vector_field.append(nn.Linear())
#                 self.vector_field.append(nn.Tanh())

#     def forward(self,z):
#         pass


class iResNetBlock(nn.Module):
    def __init(self,num_layers,hidden_size,final_activation='none',activation='tanh',n_power_iterations=1):
        self.activation = activation_func(activation)
        self.final_activation = activation_func(activation)
        self.block = nn.ModuleList()
        self.n_power_iterations = n_power_iterations

        for layer in range(num_layers):
            if layer < (num_layers - 1):
                self.block.append(
                    nn.Sequential(
                        spectral_norm(nn.Linear(hidden_size,hidden_size),self.n_power_iterations),
                        activation_func(self.activation),
                ))
            elif layer == (num_layers - 1):
                self.block.append(
                    nn.Sequential(
                        spectral_norm(nn.Linear(hidden_size,hidden_size),self.n_power_iterations),
                        activation_func(self.final_activation),
                ))


    def forward(self,z,t):
        z + self.block(z,t)

class iResNet(nn.Module):
    def __init(self):
        pass

    def forward(self):
        pass



class GRUFlow(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
