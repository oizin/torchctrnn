from .base import ODERNNBase
import torch 
import torch.nn as nn

class IMODE(nn.Module):
    """neuralJumpODECell
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    def __init__(self,UpdateNNi,UpdateNNx,ODENet,input_size_update,output_size=1,device='cpu',method='dopri5',tol={'rtol':1e-2,'atol':1e-2},options=dict(),dt_scaler=1.0):
        
        self.... = 
        
    def forward_update(self,inputx_update,inputi_update,hx_0,hi_0,h_0):
        hx_0 = self.UpdateNNx(inputx_update,hx_0,h_0)
        hi_0 = self.UpdateNNi(inputi_update,hi_0,h_0)
        return hx_0,hi_0
        
    def forward_ode(self,h,hx,hi,times,input_ode=None,n_intermediate=0):
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
        output = self.solve_ode(self.ODENet,hidden,ts)[1:]
        return output
        
        
    def forward(self,...):
        0
        
    def solve_ode(self,...):
        
        # solve in parallel
    def solve_ode(self,vector_field,h_0,time):
        """
        solve_ode
        """
        # numerical integration until next time step
        output = odeint(vector_field, h_0, time, rtol=self.tol['rtol'],atol=self.tol['atol'], method = self.method, options=self.options)
        return output    
