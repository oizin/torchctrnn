
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
