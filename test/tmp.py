import torch
import torch.nn as nn

DT_SCALER = 1/12

class FF(nn.Module):
    """FF - feedforward network
    
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim+hidden_dim, (feature_dim+hidden_dim)//2),
            nn.Tanh(),
            nn.Linear((feature_dim+hidden_dim)//2, hidden_dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,input,t,hidden):
        print(torch.cat((input,hidden),1))
        output = self.net(torch.cat((input,hidden),1))
        return output