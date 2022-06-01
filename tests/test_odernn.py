import unittest
import torch
import torch.nn as nn
from torchctrnn import NeuralODE,ODERNNCell,ODEGRUCell,ODELSTMCell,neuralJumpODECell

class TestODERNNBase(unittest.TestCase):

    def setUp(self):
        self.update_input_size = 10
        self.ode_input_size = 2
        self.hidden_size = 4
        # times
        t0 = torch.rand(1,1)
        t1 = t0 + torch.rand(1,1)
        self.times_1 = torch.cat((t0,t1),1)
        t0 = torch.rand(32,1)
        t1 = t0 + torch.rand(32,1)
        self.times_32 = torch.cat((t0,t1),1)
        t0 = torch.rand(1000,1)
        t1 = t0 + torch.rand(1000,1)
        self.times_1000 = torch.cat((t0,t1),1)
        # update inputs
        self.update_input_1 = torch.randn((1,self.update_input_size))
        self.update_input_32 = torch.randn((32,self.update_input_size))
        self.update_input_1000 = torch.randn((1000,self.update_input_size))
        # update inputs
        self.ode_input_1 = torch.randn((1,self.ode_input_size))
        self.ode_input_32 = torch.randn((32,self.ode_input_size))
        # hidden / cell state
        self.hidden_1 = torch.randn((1,self.hidden_size))
        self.hidden_32 = torch.randn((32,self.hidden_size))
        self.hidden_1000 = torch.randn((1000,self.hidden_size))
        self.cell_1 = torch.randn((1,self.hidden_size))
        self.cell_32 = torch.randn((32,self.hidden_size))

    def test_odernnbase(self):
        pass

    def test_odernn(self):
        """
        Check ODERNNCell and ODEGRUCell
        """
        check_nets = [ODERNNCell,ODEGRUCell]
        for net in check_nets:
            func = nn.Sequential(
                nn.Linear(self.hidden_size, 50),
                nn.Tanh(),
                nn.Linear(50, self.hidden_size)
            )
            odenet = NeuralODE(func,time_dependent=False,data_dependent=False)
            odernn = net(odenet,self.update_input_size,self.hidden_size)
            h_1 = odernn(self.update_input_1,self.hidden_1,self.times_1)
            self.assertIsInstance(h_1,torch.Tensor)
            self.assertEqual(h_1.size(),torch.Size([1,self.hidden_size]))
            h_32 = odernn(self.update_input_32,self.hidden_32,self.times_32)
            self.assertIsInstance(h_32,torch.Tensor)
            self.assertEqual(h_32.size(),torch.Size([32,self.hidden_size]))
            
    def test_jumpode(self):
        """
        Check neuralJumpODECell
        """
        class UpdateNet(nn.Module):
            def __init__(self,update_input_size,hidden_dim):
                super().__init__()
                self.hidden_size = hidden_dim
                self.net = nn.Sequential(
                    nn.Linear(update_input_size, 50),
                    nn.Tanh(),
                    nn.Linear(50, hidden_dim)
                )
            def forward(self,input,hidden):
                output = hidden + self.net(input)
                return output

        check_nets = [neuralJumpODECell]
        for net in check_nets:
            func = nn.Sequential(
                nn.Linear(self.hidden_size, 50),
                nn.Tanh(),
                nn.Linear(50, self.hidden_size)
            )
            jump = UpdateNet(self.update_input_size,self.hidden_size)
            odenet = NeuralODE(func,time_dependent=False,data_dependent=False)
            odernn = net(jump,odenet)
            h_1 = odernn(self.update_input_1,self.hidden_1,self.times_1)
            self.assertIsInstance(h_1,torch.Tensor)
            self.assertEqual(h_1.size(),torch.Size([1,self.hidden_size]))
            h_32 = odernn(self.update_input_32,self.hidden_32,self.times_32)
            self.assertIsInstance(h_32,torch.Tensor)
            self.assertEqual(h_32.size(),torch.Size([32,self.hidden_size]))

    def test_odelstm(self):
        """
        Check ODELSTMCell
        """
        func = nn.Sequential(
            nn.Linear(self.hidden_size, 50),
            nn.Tanh(),
            nn.Linear(50, self.hidden_size)
        )
        odenet = NeuralODE(func,time_dependent=False,data_dependent=False)
        odernn = ODELSTMCell(odenet,self.update_input_size,self.hidden_size)
        h_1,c_1 = odernn(self.update_input_1,(self.hidden_1,self.cell_1),self.times_1)
        self.assertIsInstance(h_1,torch.Tensor)
        self.assertIsInstance(c_1,torch.Tensor)
        self.assertEqual(h_1.size(),torch.Size([1,self.hidden_size]))
        self.assertEqual(c_1.size(),torch.Size([1,self.hidden_size]))
        h_32,c_32 = odernn(self.update_input_32,(self.hidden_32,self.cell_32),self.times_32)
        self.assertIsInstance(h_32,torch.Tensor)
        self.assertIsInstance(c_32,torch.Tensor)
        self.assertEqual(h_32.size(),torch.Size([32,self.hidden_size]))
        self.assertEqual(c_32.size(),torch.Size([32,self.hidden_size]))

    def test_neural_ode_arguments(self):
        """
        backend='torchdiffeq',solver='euler',atol:float=1e-3, rtol:float=1e-3,**solver_options
        """
        check_nets = [ODERNNCell,ODEGRUCell]
        for net in check_nets:
            func = nn.Sequential(
                nn.Linear(self.hidden_size, 50),
                nn.Tanh(),
                nn.Linear(50, self.hidden_size)
            )
            # set step size
            odenet = NeuralODE(func,time_dependent=False,data_dependent=False,solver='euler',solver_options={'step_size':1e-2})
            odernn = net(odenet,self.update_input_size,self.hidden_size)
            h_ = odernn(self.update_input_1000,self.hidden_1000,self.times_1000)
            self.assertFalse(torch.all(torch.isnan(h_)).item() and torch.all(torch.isnan(h_)).item())
            # tolerance
            odenet = NeuralODE(func,time_dependent=False,data_dependent=False,solver='dopri5',atol=1e-2,rtol=1e-2)
            odernn = net(odenet,self.update_input_size,self.hidden_size)
            h_ = odernn(self.update_input_1000,self.hidden_1000,self.times_1000)
            self.assertFalse(torch.all(torch.isnan(h_)).item() and torch.all(torch.isnan(h_)).item())
            # time transformation 1
            odenet = NeuralODE(func,time_func='tanh',time_dependent=False,data_dependent=False,solver='euler',solver_options={'step_size':1e-2})
            odernn = net(odenet,self.update_input_size,self.hidden_size)
            h_ = odernn(self.update_input_1000,self.hidden_1000,1e6*self.times_1000)
            self.assertFalse(torch.all(torch.isnan(h_)).item() and torch.all(torch.isnan(h_)).item())
            # time transformation 2
            time_func = lambda x : x*1e-2
            odenet = NeuralODE(func,time_func=time_func,time_dependent=False,data_dependent=False,solver='euler',solver_options={'step_size':1e-2})
            odernn = net(odenet,self.update_input_size,self.hidden_size)
            h_ = odernn(self.update_input_1000,self.hidden_1000,1e2*self.times_1000)
            self.assertFalse(torch.all(torch.isnan(h_)).item() and torch.all(torch.isnan(h_)).item())

    def test_large_time_gaps(self):
        pass

