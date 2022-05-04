import unittest
import torch
import torch.nn as nn

# class TestODERNNBase(unittest.TestCase):

#     def setUp(self) -> None:
#         self.update_input_size = 10
#         self.ode_input_size = 2
#         self.hidden_size = 4
#         self.ntimesteps = 10
#         # times
#         t0 = torch.rand(1,self.ntimesteps,1)
#         t1 = t0 + torch.rand(1,self.ntimesteps,1)
#         self.times_1 = torch.cat((t0,t1),1)
#         t0 = torch.rand(32,self.ntimesteps,1)
#         t1 = t0 + torch.rand(32,self.ntimesteps,1)
#         self.times_32 = torch.cat((t0,t1),1)
#         # update inputs
#         self.update_input_1 = torch.randn((1,self.ntimesteps,self.update_input_size))
#         self.update_input_32 = torch.randn((32,self.ntimesteps,self.update_input_size))
#         # update inputs
#         self.ode_input_1 = torch.randn((1,self.ntimesteps,self.ode_input_size))
#         self.ode_input_32 = torch.randn((32,self.ntimesteps,self.ode_input_size))

#     def test_odernnbase(self):
#         pass