import unittest
import torch
import torch.nn as nn
from torchctrnn.data.simulations import GlucoseData

class TestGLucose(unittest.TestCase):

    def test_seed(self):
        sim1 = GlucoseData()
        df1 = sim1.simulate(1,seed=1234)
        df2 = sim1.simulate(1,seed=1234)
        sim2 = GlucoseData()
        df3 = sim2.simulate(1,seed=1234)
        sim3 = GlucoseData(nonstationary=1)
        df4 = sim3.simulate(1,seed=1234)

        self.assertEqual(df1.glucose_t[0], df2.glucose_t[0])
        self.assertEqual(df1.glucose_t[0], df3.glucose_t[0])

if __name__ == '__main__':
    unittest.main()
