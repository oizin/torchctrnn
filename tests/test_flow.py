from os import times_result
import unittest
import torch
import torch.nn as nn
from torchctrnn import ResNetFlow,NeuralFlow

class TestResNetFlow(unittest.TestCase):

    def setUp(self):
        self.input_size = 10
        self.hidden_size = 4
        # hidden states
        self.h_1 = torch.randn((1,self.hidden_size))
        self.h_32 = torch.randn((32,self.hidden_size))
        t0 = torch.rand(1,1)
        t1 = t0 + torch.rand(1,1)
        self.times_1 = torch.cat((t0,t1),1)
        t0 = torch.rand(32,1)
        t1 = t0 + torch.rand(32,1)
        self.times_32 = torch.cat((t0,t1),1)
        # exogeneous inputs
        self.input_1 = torch.randn((1,self.input_size))
        self.input_32 = torch.randn((32,self.input_size))

    def test_resnet_flow(self):
        # ODENetfromSequential approach
        resnetflow = ResNetFlow(self.hidden_size,50)
        neuralflow = NeuralFlow(resnetflow)
        # self.assertIsInstance(odenet.forward(hidden=self.h_1),torch.Tensor)
        # self.assertIsInstance(odenet.forward(hidden=self.h_32),torch.Tensor)
        # test output (general)
        self.assertIsInstance(neuralflow(self.h_1,self.times_1[:,1:2]-self.times_1[:,0:1]),torch.Tensor)
        self.assertIsInstance(neuralflow(self.h_32,self.times_32[:,1:2]-self.times_32[:,0:1]),torch.Tensor)
        # test output (dimensions)
        self.assertEqual(neuralflow(self.h_1,self.times_1[:,1:2]-self.times_1[:,0:1]).shape, torch.Size([1,self.hidden_size]))
        self.assertEqual(neuralflow(self.h_32,self.times_32[:,1:2]-self.times_32[:,0:1]).shape, torch.Size([32,self.hidden_size]))

if __name__ == '__main__':
    unittest.main()
