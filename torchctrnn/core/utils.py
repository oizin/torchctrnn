
import torch
import torch.nn as nn

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['tanh', nn.Tanh()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
        ['selu', nn.SELU()],
        ['none', nn.Identity()]
    ])[activation]

def time_func(func):
    return  nn.ModuleDict([
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])[func]
