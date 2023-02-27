import torch 
from torch import Tensor
import torch.nn as nn
import inspect
from typing import Tuple, Callable,Union

def odeint_batch(vector_field,hidden,limits,method,options):
    assert method == 'euler'

    dt = options['step_size']
    steps = torch.floor((limits[:,1] - limits[:,0]) / dt)
    ht = hidden
    max_steps = torch.max(steps)
    for step in range(0,int(max_steps)):
        m = (step < steps)
        m = m.reshape((m.numel(), 1))
        t_ = limits[:,0] + step*dt 
        t_ = t_.unsqueeze(1)
        ht = ht + dt*vector_field(t_,ht) * m
    return ht
