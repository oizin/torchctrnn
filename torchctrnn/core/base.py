import torch 
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import inspect

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
                
class ODENetBase(nn.Module):
    def __init__(self,ODENet,dt_scaler):
        super(ODENetBase, self).__init__()
        self.dt_scaler=dt_scaler
        self.calls=0
        self.track_calls=False
        
        args = inspect.getfullargspec(ODENet.forward)[0]
        if args != ['self', 'input', 't', 'dt', 'hidden']:
            raise NameError("ODENet's forward method should have arguments: 'input', 't' 'dt' and 'hidden' (in that order)")
#         if 'hidden_size' not in dir(ODENet):
#             raise ValueError("ODENet should have attribute hidden_size")
        
        self.net = ODENet
        self.input_ode = torch.zeros(1,1)
        self.times = torch.zeros(1,1)
        self.time_gaps = torch.zeros(1,1)
                
    def forward(self,t,hidden):
        if self.track_calls == True:
            self.calls += 1
        
        output = self.net(self.input_ode,self.times[:,0]+t.reshape(1,1)*self.time_gaps,self.time_gaps,hidden)*self.dt_scaler*self.time_gaps
        return output

class NeuralFlowBase(nn.Module):
    def __init__(self,NeuralFlow):
        
        args = inspect.getfullargspec(NeuralFlow.forward)[0]
        if args != ['self', 'input', 't', 'dt', 'hidden']:
            raise NameError("NeuralFlow's forward method should have arguments: 'input', 't' 'dt' and 'hidden' (in that order)")
        
        self.flow = NeuralFlow
        self.input_ode = torch.zeros(1,1)
        self.times = torch.zeros(1,1)
        self.time_gaps = torch.zeros(1,1)
                
    def forward(self,t,hidden):
        output = self.net(self.input_ode,self.times[:,0]+t.reshape(1,1)*self.time_gaps,self.time_gaps,hidden)
        return output

class CTRNNBase(nn.Module):

    def __init__(self,UpdateNN,device):
        super(CTRNNBase,self).__init__()

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

class ODERNNBase(CTRNNBase):
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
    
    def __init__(self,UpdateNN,ODENet,device='cpu',output_size=1,method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        super(ODERNNBase,self).__init__(UpdateNN,device)
        
        self.ODENet = ODENetBase(ODENet,dt_scaler)
        self.method = method
        self.options = options
        self.tol=tol
        self.dt_scaler=dt_scaler
                    
    def forward_ode(self,hidden,times,input_ode=None,n_intermediate=0):
        """
        forward_ode
        """
        # enable input and time_gaps to be passed to ODENet.forward
        self.ODENet.input_ode = input_ode
        self.ODENet.time_gaps = times[:,1:2] - times[:,0:1]
        if n_intermediate > 1:
            ts = torch.linspace(0,1,2+n_intermediate)
        else:
            ts = torch.tensor([0,1.0])
        output = self.solve_ode(self.ODENet,hidden,ts)
        return output
    
    def solve_ode(self,vector_field,h_0,time):
        """
        solve_ode
        """
        # numerical integration until next time step
        # torchdiffeq backend
        output = odeint(vector_field, h_0, time, rtol=self.tol['rtol'],atol=self.tol['atol'], method = self.method, options=self.options)
        output = output[1:].squeeze(0)
        return output    
    
class FlowRNNBase(CTRNNBase):
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
    
    def __init__(self,UpdateNN,NeuralFlow,device='cpu',output_size=1,method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        super(FlowRNNBase,self).__init__(UpdateNN,device)
        
        self.vector_field = NeuralFlowBase(NeuralFlow)
        self.method = method
        self.options = options
        self.tol=tol
        self.dt_scaler=dt_scaler
                
    def forward_ode(self,hidden,times,input_ode=None,n_intermediate=0):
        """
        forward_ode
        """
        # enable input and time_gaps to be passed to the ODE
        self.ODENet.input_ode = input_ode
        self.ODENet.time_gaps = times[:,1:2] - times[:,0:1]
        output = self.vector_field.forward(hidden,times)
        return output