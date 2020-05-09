from contrast_loss_utils import supervised_nt_xent_loss
from supcontrast import SupConLoss
import torch
import tensorflow as tf
import unittest
import numpy as np
np.random.seed(42)


class TestSupContrastLoss(unittest.TestCase):
    '''To test my tensorflow implementation of Supervised Contrastive loss yields the same
    values with the Torch implementation.
    '''

    def setUp(self):
        X = np.random.randn(32, 128)
        X /= np.linalg.norm(X, axis=0)
        self.X = X.astype(np.float32)
        self.y = np.random.choice(np.arange(3), 32, replace=True)
        self.temperature = 0.5
        self.base_temperature = 0.07

    def test_nt_xent_loss_equals_sup_con_loss(self):
        l1 = supervised_nt_xent_loss(tf.constant(self.X),
                                     tf.constant(self.y),
                                     temperature=self.temperature,
                                     base_temperature=self.base_temperature
                                     )

        scl = SupConLoss(temperature=self.temperature,
                         base_temperature=self.base_temperature
                         )
        l2 = scl.forward(features=torch.Tensor(self.X.reshape(32, 1, 128)),
                         labels=torch.Tensor(self.y)
                         )
        print('l1 = {}'.format(l1.numpy()))
        print('l2 = {}'.format(l2.numpy()))
        self.assertTrue(np.allclose(l1.numpy(), l2.numpy()))


if __name__ == "__main__":
    unittest.main()
