from .base import ODERNNBase
import torch 
import torch.nn as nn

class CDENetBase(nn.Module):
    """CDENetBase
    Args:
        ....
    """
    def __init__(self,CDENet):
        super(CDENetBase, self).__init__()
        
        self.net = CDENet
        self.input_ode = torch.zeros(1,1)
        self.time_gaps = torch.zeros(1,1)

    def forward(self,t,hidden):
        output = self.net(t.reshape(1,1)*self.time_gaps,hidden) * self.time_gaps.unsqueeze(1)
        return torch.matmul(output,self.input_ode.unsqueeze(2)).squeeze(2)
    

class neuralCDECell(ODERNNBase):
    """neuralCDECell. Rectilinear.
    
    Args:
    
    Structure of CDENet:

    Return:
        Tensor
    """
    
    def __init__(self,CDENet,output_size=1, device='cpu'):
        super(ODERNNBase,self).__init__()
        
        self.ODENet = CDENetBase(CDENet)
        self.device = device
                
    def forward(self,input_t1,h_0,times,input_t0=None):   
        """
        forward
        
        """
        # calculate dX and dt
        if input_t0 == None:
            dinput = torch.zeros_like(input_t1)
        else:
            dinput = input_t1 - input_t0
        dt = times[:,1:2] - times[:,0:1]
        dt_z = torch.cat((dt,torch.zeros_like(dinput)),1)
        dinput_z = torch.cat((torch.zeros_like(dt),dinput),1)
        
        # use ODENet to update as new information recieved
        hidden = self.forward_update(dinput_z,h_0)
        # use ODENet to 'evolve' state to next timestep
        output = self.forward_ode(hidden,times,dinput_z)
        return output
    
    def forward_update(self,input_update,h_0):
        """
        forward_update
        """
        output = self.forward_ode(h_0,torch.tensor([0,1.0]).expand(input_update.size(0),2),input_update)
        return output
    
#     def forward_ode(self,hidden,times,input_ode=None):
#         """
#         forward_ode
#         -----> use for predicting a trajectory
#         """
#         # enable input and time_gaps to be passed to ODENet.forward
#         self.ODENet.input_ode = input_ode
#         self.ODENet.time_gaps = times[:,1:2] - times[:,0:1]
#         output = self.solve_ode(self.CDENet,hidden,torch.tensor([0,1.0]).to(self.device))[1]
#         return output
