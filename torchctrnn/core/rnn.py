from .base import ODERNNBase
import torch 
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class neuralJumpODECell(ODERNNBase):
    """neuralJumpODECell
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    def __init__(self,UpdateNN,ODENet,input_size_update,output_size=1,device='cpu',method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        ODERNNBase.__init__(self,UpdateNN,ODENet,output_size,device,method,tol,options,dt_scaler)
            
class ODERNNCell(ODERNNBase):
    """
    ODERNNCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu',method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        hidden_size = ODENet.hidden_size
        rnn = nn.RNNCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device,method,tol,options,dt_scaler)
        
class ODEGRUCell(ODERNNBase):
    """
    ODEGRUCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu',method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        hidden_size = ODENet.hidden_size
        rnn = nn.GRUCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device,method,tol,options,dt_scaler)
        
class ODELSTMCell(ODERNNBase):
    """
    ODELSTMCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu',method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        hidden_size = ODENet.hidden_size
        rnn = nn.LSTMCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device,method,tol,options,dt_scaler)
        
    def forward_update(self,input_update : Tensor,h_0 : Tuple[Tensor,Tensor]) -> Tensor:
        """
        forward_update
        """
        output = self.updateNN(input_update,h_0)
        return output
    
    def forward(self,input_update : Tensor,h_0 : Tuple[Tensor,Tensor],times,input_ode=None,n_intermediate=0) -> Tuple[Tensor,Tensor]:   
        """ 
        forward
        """
        if (type(h_0) != tuple):
            raise ValueError("h_0 should be a tuple of (hidden state, cell state)")
        # discrete update/jump as new information receieved
        hidden,cell = self.forward_update(input_update,h_0)
        # use ODENet to 'evolve' hidden state (but not cell state) to next timestep
        output = self.forward_ode(hidden,times,input_ode,n_intermediate)
        return output,cell
