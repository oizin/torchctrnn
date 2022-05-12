import torch 
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import inspect
from typing import Tuple, Callable,Union
from .ctrnn import _CTRNNBase
from .odenet import NeuralODE

# Notes
# training data of shape: (batch,seq,features)
# this is for one time so data of shape (batch,features), time index dropped

class _VectorField(nn.Module):

    def __init__(self,NeuralODE,delta_t,times:Tensor=None,input:Tensor=None):
        """
        TODO
        """
        super(_VectorField, self).__init__()
        self.input = input
        self.NeuralODE = NeuralODE
        self.times = times
        self.has_input_arg = NeuralODE.has_input_arg
        self.has_t_arg = NeuralODE.has_t_arg
        self.delta_t = delta_t

    def forward(self,t:Tensor,z:Tensor):
        """
        Args:
            z : hidden state. Dimensions are (TODO,TODO)
            t : time variable used in ODESolve. Dimensions are torch.Size([1]).
        """
        # conditional evaluation dependent on f=NeuralODE arguments
        if self.has_t_arg:
            t = t.reshape(1,1)
            t_ = self.times[:,0].unsqueeze(1)+t*self.delta_t
            if self.has_input_arg:
                # dh/ds=f(h,t,x)
                output = self.NeuralODE(hidden=z,t=t_,input=self.input)
            else:
                # dh/ds=f(h,t)
                output = self.NeuralODE(hidden=z,t=t_)
        elif self.has_input_arg:
            # dh/ds=f(h,x)
            output = self.NeuralODE(hidden=z,input=self.input)
        else:
            # dh/ds=f(h)
            output = self.NeuralODE(hidden=z)
        # dh/dt
        output = output*self.delta_t
        return output


class _ODERNNBase(_CTRNNBase):
    """Base class for ODE recurrent neural network (RNN) (e.g. vanilla RNNs, Jump NNs, GRUs and LSTMs)
    
    Args:
        NeuralODE (nn.Module): The neural network
        UpdateNN (nn.Module): The neural network
    
    Structure of NeuralODE:
            output_size: dimension of output
            input_update_size: dimension of update features (can be larger than output)
            input_ode_size: dimension of features you wish to pass to NeuralODE
            hidden_size: dimension of hidden state

    Structure of UpdateNN:
            output_size: dimension of output
            input_update_size: dimension of update features (can be larger than output)
            input_ode_size: dimension of features you wish to pass to NeuralODE
            hidden_size: dimension of hidden state
            
    Return:
        Tensor
    """
    
    def __init__(self,UpdateNN:nn.Module,NeuralODE:NeuralODE,device='cpu'):
        super(_ODERNNBase,self).__init__(UpdateNN,device)
        
        self.NeuralODE = NeuralODE
                    
    def forward_ode(self,hidden:Tensor,times:Tensor,input_ode:Tensor=None,n_intermediate:int=0) -> Tensor:
        """
        forward_ode
        """
        delta_t = times[:,1:2] - times[:,0:1]
        # dh/dt = dh/ds(h,...)*dt
        vector_field = _VectorField(self.NeuralODE,delta_t,times,input_ode)
        # TODO: next few lines are terrible - vary depending on if batch_size=1!
        if n_intermediate > 1:  
            limits = torch.linspace(0,1,2+n_intermediate)
        else:
            limits = torch.tensor([0,1.0])
        output = self.solve_ode(vector_field,hidden,limits)
        return output
    
    def solve_ode(self,vector_field:_VectorField,hidden:Tensor,limits:Tensor):
        """
        solve_ode
        """
        # numerical integration until next time step
        # torchdiffeq backend
        output = odeint(vector_field, hidden, limits,
                        method=self.NeuralODE.solver,
                        atol=self.NeuralODE.atol,rtol=self.NeuralODE.rtol,
                        options=self.NeuralODE.solver_options)
        output = output[1:].squeeze(0)
        return output    
    
