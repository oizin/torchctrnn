import torch 
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import inspect
from typing import Tuple, Callable

# Notes
# training data of shape: (batch,seq,features)
# this is for one time so data of shape (batch,features), time index dropped

class _VectorField(nn.Module):

    def __init__(self,ODENet,delta_t,times:Tensor=None,input:Tensor=None,has_input_arg:bool=False,has_t_arg:bool=False):
        """
        
        
        """
        super(_VectorField, self).__init__()
        self.input = input
        self.ODENet = ODENet
        self.times = times
        self.has_input_arg = ODENet.has_input_arg
        self.has_t_arg = ODENet.has_t_arg
        self.delta_t = delta_t

    def forward(self,t:Tensor,z:Tensor):
        """
        Args:
            z : hidden state. Dimensions are (TODO,TODO)
            t : time variable used in ODESolve. Dimensions are torch.Size([1]).
        """
        # conditional evaluation dependent on f=ODENet arguments
        if self.has_t_arg:
            t = t.reshape(1,1)
            t_ = self.times[:,0]+t*self.delta_t
            if self.has_input_arg:
                # dh/ds=f(h,t,x)
                output = self.ODENet(self.input,t_,z)
            else:
                # dh/ds=f(h,t,x)
                output = self.ODENet(self.input,t_,z)
        elif self.has_input_arg:
            # dh/ds=f(h,x)
            output = self.ODENet(self.input,z)
        else:
            # dh/ds=f(h)
            output = self.ODENet(z)
        # dh/dt
        output = output*self.delta_t
        return output

class _ODENetBase(nn.Module):
    """
    purpose of this class?
    """
    def __init__(self,ODENet):
        super(_ODENetBase, self).__init__()
        self.has_input_arg = False
        self.has_t_arg = False

        # check what arguments ODENet (the vector field) takes:
        args = inspect.getfullargspec(ODENet.forward)[0]
        check_args = ['input', 't', 'hidden']
        check_res = [chk for chk in check_args if chk not in args]
        if len(check_res) > 0:
            raise Warning("ODENet's forward method missing args: {}. These are assumed XYZ".format(check_res))
        if 'input' in args:
            self.has_input_arg = True
        if 't' in args:
            self.has_t_arg = True
        self.net = ODENet
                
class ODENet(nn.Module):

    def __init__(self,net,method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),backend='torchdiffeq'):
        super.__init__()

        self.method = method
        self.options = options
        self.tol=tol
        self.backend=backend
        self.has_t_arg
        self.has_input_arg

class _ODERNNBase(_CTRNNBase):
    """Base class for ODE recurrent neural network (RNN) (e.g. vanilla RNNs, Jump NNs, GRUs and LSTMs)
    
    Args:
        ODENet (nn.Module): The neural network
        UpdateNN (nn.Module): The neural network
    
    Structure of ODENet:
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
    
    def __init__(self,UpdateNN,ODENet,device='cpu'):
        super(_ODERNNBase,self).__init__(UpdateNN,device)
        
        self.ODENet = _ODENetBase(ODENet)
                    
    def forward_ode(self,hidden,times,input_ode=None,n_intermediate=0):
        """
        forward_ode
        """
        delta_t = times[:,1:2] - times[:,0:1]
        # dh/dt = dh/ds(h,...)*dt
        vector_field = _VectorField(self.ODENet,delta_t,times,input_ode)
        # TODO: next few lines are terrible!
        if n_intermediate > 1:  
            limits = torch.linspace(0,1,2+n_intermediate)
        else:
            limits = torch.tensor([0,1.0])
        output = self.solve_ode(vector_field,hidden,limits)
        return output
    
    def solve_ode(self,vector_field,hidden,limits):
        """
        solve_ode
        """
        # numerical integration until next time step
        # torchdiffeq backend
        output = odeint(vector_field, hidden, limits,
                        rtol=self.tol['rtol'],atol=self.ODENet.tol['atol'],
                        method = self.ODENet.method,options=self.ODENet.options)
        output = output[1:].squeeze(0)
        return output    
    
