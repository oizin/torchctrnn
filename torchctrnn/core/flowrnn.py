import torch 
from torch import Tensor
import torch.nn as nn
import inspect
from typing import Tuple, Callable
from .ctrnn import _CTRNNBase

class _NeuralFlowBase(nn.Module):
    def __init__(self,NeuralFlow):
        super(_NeuralFlowBase, self).__init__()
        
        args = inspect.getfullargspec(NeuralFlow.forward)[0]
        if args != ['self', 'input', 't', 'dt', 'hidden']:
            raise NameError("NeuralFlow's forward method should have arguments: 'input', 't' and 'hidden' (in that order)")
        
        self.flow = NeuralFlow
        self.input_ode = torch.zeros(1,1)
        self.times = torch.zeros(1,1)
        self.time_gaps = torch.zeros(1,1)
                
    def forward(self,t,hidden):
        output = self.net(self.input_ode,self.times[:,0]+t.reshape(1,1)*self.time_gaps,self.time_gaps,hidden)
        return output

class _FlowRNNBase(_CTRNNBase):
    """Base class for Flow recurrent neural network (RNN) (e.g. vanilla RNNs, Jump NNs, GRUs and LSTMs)
    
    Args:
        NeuralFlow (nn.Module): The *solution* to an ordinary differential equation
        UpdateNN (nn.Module): The neural network
    
    Structure of NeuralFlow:
            output_size: dimension of output
            input_update_size: dimension of update features (can be larger than output)
            input_ode_size: dimension of features you wish to pass to ODENet
            hidden_size: dimension of hidden state

    Structure of UpdateNN:
            output_size: dimension of output
            input_update_size: dimension of update features (can be larger than output)
            input_ode_size: dimension of features you wish to pass to ODENet
            hidden_size: dimension of hidden state
            
    Return:
        Tensor
    """
    
    def __init__(self,UpdateNN,NeuralFlow,device='cpu'):
        super(_FlowRNNBase,self).__init__(UpdateNN,device)
        
        self.vector_field = _NeuralFlowBase(NeuralFlow)
                
    def forward_ode(self,hidden:Tensor,times:Tensor,input_ode:Tensor=None,n_intermediate:int=0) -> Tensor:
        """
        forward_ode
        """
        # enable input and time_gaps to be passed to the ODE
        output = self.vector_field.forward(hidden,times,input_ode)
        return output