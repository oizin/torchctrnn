from os import times_result
import unittest
import torch
import torch.nn as nn
from torchctrnn import NeuralODE

class TestNeuralODE(unittest.TestCase):

    def setUp(self):
        self.input_size = 10
        self.hidden_size = 4
        # hidden states
        self.h_1 = torch.randn((1,self.hidden_size))
        self.h_32 = torch.randn((32,self.hidden_size))
        # times
        self.times_1 = torch.rand(1,1)
        self.times_32 = torch.rand(32,1)
        # exogeneous inputs
        self.input_1 = torch.randn((1,self.input_size))
        self.input_32 = torch.randn((32,self.input_size))

    def test_node_seq_hidden(self):
        # ODENetfromSequential approach
        func = nn.Sequential(
            nn.Linear(self.hidden_size, 50),
            nn.Tanh(),
            nn.Linear(50, self.hidden_size)
        )
        odenet = NeuralODE(func,time_dependent=False,data_dependent=False)
        print(odenet)
        # self.assertIsInstance(odenet.forward(hidden=self.h_1),torch.Tensor)
        # self.assertIsInstance(odenet.forward(hidden=self.h_32),torch.Tensor)
        self.assertIsInstance(odenet.forward(self.h_1),torch.Tensor)
        self.assertIsInstance(odenet.forward(self.h_32),torch.Tensor)
    
    def test_node_seq_hidden_t(self):
        # ODENetfromSequential approach
        func = nn.Sequential(
            nn.Linear(self.hidden_size+1, 50),
            nn.Tanh(),
            nn.Linear(50, self.hidden_size)
        )
        odenet = NeuralODE(func,time_dependent=True,data_dependent=False)
        self.assertIsInstance(odenet.forward(hidden=self.h_1,t=self.times_1),torch.Tensor)
        self.assertIsInstance(odenet.forward(hidden=self.h_32,t=self.times_32),torch.Tensor)
        self.assertIsInstance(odenet.forward(self.h_1,self.times_1),torch.Tensor)
        self.assertIsInstance(odenet.forward(self.h_32,self.times_32),torch.Tensor)

    def test_node_hidden(self):
        # ODENet approach
        class Func(nn.Module):
            def __init__(self,hidden_size):
                super().__init__()

                self.hidden_size = hidden_size
                self.net = nn.Sequential(
                    nn.Linear(hidden_size, 50),
                    nn.Tanh(),
                    nn.Linear(50, hidden_size),
                )

            def forward(self,hidden):
                return self.net(hidden)
        odenet = NeuralODE(Func(self.hidden_size))
        self.assertIsInstance(odenet.forward(hidden=self.h_1),torch.Tensor)
        self.assertIsInstance(odenet.forward(hidden=self.h_32),torch.Tensor)
        self.assertIsInstance(odenet.forward(self.h_1),torch.Tensor)
        self.assertIsInstance(odenet.forward(self.h_32),torch.Tensor)


if __name__ == '__main__':
    unittest.main()