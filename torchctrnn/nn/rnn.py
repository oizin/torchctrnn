from ..core.odernn import _ODERNNBase
from ..core.ctrnn import _CTRNNBase
from ..core.odenet import NeuralODE
from ..core.flownet import NeuralFlow
from ..core.flowrnn import _FlowRNNBase
#from ..core.flownet import NeuralFlow

import torch 
from torch import Tensor
import torch.nn as nn
from typing import Tuple,Union

class neuralJumpODECell(_ODERNNBase):
    """neuralJumpODECell
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    def __init__(self,UpdateNN,ODENet):
        _ODERNNBase.__init__(self,UpdateNN,ODENet)

class CTRNNCell(_CTRNNBase):
    """
    CTRNNCell

    TODO
    """
    def __init__(self,VectorField:Union[NeuralODE,NeuralFlow],input_size_update:int,hidden_size:int):
        rnn = nn.RNNCell(input_size_update,hidden_size)
        assert isinstance(VectorField,(NeuralODE,NeuralFlow))
        if isinstance(VectorField,NeuralODE):
            _ODERNNBase.__init__(self,rnn,VectorField)
        elif isinstance(VectorField,NeuralFlow):
            _FlowRNNBase.__init__(self,rnn,VectorField)

class ODERNNCell(_ODERNNBase):
    """
    ODERNNCell

    TODO

    Example:  
    ```
    func = nn.Sequential(  
        nn.Linear(4, 50),  
        nn.Tanh(),  
        nn.Linear(50,4)  
    )
    odenet = NeuralODE(func,time_dependent=False,data_dependent=False)  
    odernn = ODERNNCell(odenet,10,4)  
    odernn(input_update,hidden,times)  
    ```
    """
    def __init__(self,NeuralODE,input_size_update:int,hidden_size:int):
        rnn = nn.RNNCell(input_size_update,hidden_size)
        _ODERNNBase.__init__(self,rnn,NeuralODE)
        
class ODEGRUCell(_ODERNNBase):
    """
    ODEGRUCell
    """
    def __init__(self,NeuralODE,input_size_update:int,hidden_size:int):
        rnn = nn.GRUCell(input_size_update,hidden_size)
        _ODERNNBase.__init__(self,rnn,NeuralODE)

class FlowGRUCell(_FlowRNNBase):
    """
    ODEGRUCell
    """
    def __init__(self,NeuralFlow,input_size_update:int,hidden_size:int):
        rnn = nn.GRUCell(input_size_update,hidden_size)
        _FlowRNNBase.__init__(self,rnn,NeuralFlow)

class ODELSTMCell(_ODERNNBase):
    """
    ODELSTMCell
    """
    def __init__(self,NeuralODE,input_size_update:int,hidden_size:int):
        rnn = nn.LSTMCell(input_size_update,hidden_size)
        _ODERNNBase.__init__(self,rnn,NeuralODE)
        
    def forward_update(self,input_update : Tensor,hidden : Tuple[Tensor,Tensor]) -> Tensor:
        """
        forward_update
        """
        output = self.updateNN(input_update,hidden)
        return output
    
    def forward(self,input_update : Tensor,hidden : Tuple[Tensor,Tensor],times,input_ode=None,n_intermediate=0) -> Tuple[Tensor,Tensor]:   
        """ 
        forward
        """
        if (type(hidden) != tuple):
            raise ValueError("h_0 should be a tuple of (hidden state, cell state)")
        # discrete update/jump as new information receieved
        hidden,cell = self.forward_update(input_update,hidden)
        # use ODENet to 'evolve' hidden state (but not cell state) to next timestep
        output = self.forward_ode(hidden,times,input_ode,n_intermediate)
        return output,cell
