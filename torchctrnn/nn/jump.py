from .odernn import _ODERNNBase
import torch 
import torch.nn as nn

class JumpODECell(_ODERNNBase):
    """JumpODECell
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    def __init__(self,UpdateNN,ODENet,output_size=1,method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict()):
        _ODERNNBase.__init__(self,UpdateNN,ODENet,output_size,method,tol,options,dt_scaler)
        
        
    def forward_update(self,input_update):
        """
        forward_update
        
        Encoder
        """
        output = self.updateNN(input_update)
        return output
    
    def forward(self,input_update,h_0,times,input_ode=None,n_intermediate=0):   
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

