from .base import ODERNNBase
import torch 
import torch.nn as nn

class LatentJumpODECell(ODERNNBase):
    """LatentJumpODECell
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    def __init__(self,UpdateNN,ODENet,output_size=1,device='cpu'):
        ODERNNBase.__init__(self,UpdateNN,ODENet,output_size,device)
    
class ODERNNCell(ODERNNBase):
    """
    ODERNNCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu'):
        hidden_size = ODENet.hidden_size
        rnn = nn.RNNCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device)
        
class ODEGRUCell(ODERNNBase):
    """
    ODEGRUCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu'):
        hidden_size = ODENet.hidden_size
        rnn = nn.GRUCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device)
        
class ODELSTMCell(ODERNNBase):
    """
    ODELSTMCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu'):
        hidden_size = ODENet.hidden_size
        rnn = nn.LSTMCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device)
        
    def forward_update(self,input_update,h_0):
        """
        forward_update
        """
        output = self.updateNN(input_update,h_0)
        return output
    
    def forward(self,input_update,h_0,times,input_ode=None):   
        """ 
        forward
        """
        if (type(h_0) != tuple):
            raise ValueError("h_0 should be a tuple of (hidden state, cell state)")
        # discrete update/jump as new information receieved
        hidden,cell = self.forward_update(input_update,h_0)
        # use ODENet to 'evolve' hidden state (but not cell state) to next timestep
        output = self.forward_ode(hidden,times,input_ode)
        return output,cell
