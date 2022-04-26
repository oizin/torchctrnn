import unittest
import torch
import torchctrnn

class TestODENet(unittest.TestCase):
    def pass_kwargs_to_odenet(self):
        3

    def test_rnn_batch_1(self):
        x = torch.randn(1,10,10)

    def test_rnn_batch_32(self):
        x = torch.randn(32,10,10)



if __name__ == '__main__':
    unittest.main()