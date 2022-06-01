from os import times_result
import unittest
import torch
import torch.nn as nn
from torchctrnn import ResNetFlow,NeuralFlow,FlowGRUCell,FlowLSTMCell

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
        # hidden / cell state
        self.hidden_1 = torch.randn((1,self.hidden_size))
        self.hidden_32 = torch.randn((32,self.hidden_size))
        self.hidden_1000 = torch.randn((1000,self.hidden_size))
        self.cell_1 = torch.randn((1,self.hidden_size))
        self.cell_32 = torch.randn((32,self.hidden_size))


    def test_resnet_flow(self):
        # nets
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


class TestFlowRNN(unittest.TestCase):

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

    def test_flowrnn(self):
        """
        Check and ODEGRUCell
        """
        check_nets = [FlowGRUCell]
        for net in check_nets:
            # nets
            resnetflow = ResNetFlow(self.hidden_size,50)
            neuralflow = NeuralFlow(resnetflow)
            rnnflow = net(neuralflow,self.update_input_size,self.hidden_size)
            h_1 = rnnflow(self.update_input_1,self.hidden_1,self.times_1)
            self.assertIsInstance(h_1,torch.Tensor)
            self.assertEqual(h_1.size(),torch.Size([1,self.hidden_size]))
            h_32 = rnnflow(self.update_input_32,self.hidden_32,self.times_32)
            self.assertIsInstance(h_32,torch.Tensor)
            self.assertEqual(h_32.size(),torch.Size([32,self.hidden_size]))

    def test_flowlstm(self):
        """
        Check FlowLSTMCell
        """
        resnetflow = ResNetFlow(self.hidden_size,50)
        neuralflow = NeuralFlow(resnetflow)
        odernn = FlowLSTMCell(neuralflow,self.update_input_size,self.hidden_size)
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


if __name__ == '__main__':
    unittest.main()
