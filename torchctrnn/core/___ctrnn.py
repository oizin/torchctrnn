import torch 
from torch import Tensor
import torch.nn as nn
import inspect
from typing import Tuple, Callable

# Notes
# training data of shape: (batch,seq,features)
# this is for one time so data of shape (batch,features), time index dropped

class UpdateNNBase(nn.Module):
    def __init__(self,UpdateNN):
        super(UpdateNNBase, self).__init__()

        args = inspect.getfullargspec(UpdateNN.forward)[0]
        if (args != ['self', 'input', 'hidden']) and (args != ['self', 'input', 'hx']):
            raise NameError("UpdateNN's forward method should have arguments: 'input' and 'hidden' or 'input' and 'hx' (in that order)")
#         if 'hidden_size' not in dir(UpdateNN):
#             raise ValueError("UpdateNN should have attribute hidden_size")

        self.net = UpdateNN
    
    def forward(self,input,hidden):
        """function with types documented in the docstring.
        
        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            The return value. True for success, False otherwise.
        """
        output = self.net(input,hidden)
        return output

class _CTRNNBase(nn.Module):

    def __init__(self,UpdateNN,device):
        super(_CTRNNBase,self).__init__()

        self.updateNN = UpdateNNBase(UpdateNN)
        self.device = device

    def forward(self,input_update,h_0,times,input_ode=None,n_intermediate=0):
        """forward
        
        Args:
            times (): 2d t0 and t1
        """
        # discrete update/jump as new information receieved
        hidden = self.forward_update(input_update,h_0)
        # use vector field to 'evolve' hidden state
        output = self.forward_ode(hidden,times,input_ode,n_intermediate)
        return output

    def forward_update(self,input_update,h_0):
        """
        forward_update
        """
        output = self.updateNN(input_update,h_0)
        return output