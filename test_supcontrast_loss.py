from losses import supervised_nt_xent_loss
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
        self.batch_size = 128
        X = np.random.randn(self.batch_size, 128)
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        self.X = X.astype(np.float32)
        self.y = np.random.choice(np.arange(10), self.batch_size, replace=True)

        # very small batch where there could be only class with only one example
        self.batch_size_s = 8
        X_s = np.random.randn(self.batch_size_s, 128)
        X_s /= np.linalg.norm(X_s, axis=1).reshape(-1, 1)
        self.X_s = X_s.astype(np.float32)
        self.y_s = np.random.choice(
            np.arange(10), self.batch_size_s, replace=True)

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
        l2 = scl.forward(features=torch.Tensor(self.X.reshape(self.batch_size, 1, 128)),
                         labels=torch.Tensor(self.y)
                         )
        print('\nLosses from normal batch size={}:'.format(self.batch_size))
        print('l1 = {}'.format(l1.numpy()))
        print('l2 = {}'.format(l2.numpy()))
        self.assertTrue(np.allclose(l1.numpy(), l2.numpy()))

    def test_nt_xent_loss_and_sup_con_loss_small_batch(self):
        # on very small batch, the SupConLoss would return NaN
        # whereas supervised_nt_xent_loss will ignore those classes
        l1 = supervised_nt_xent_loss(tf.constant(self.X_s),
                                     tf.constant(self.y_s),
                                     temperature=self.temperature,
                                     base_temperature=self.base_temperature
                                     )

        scl = SupConLoss(temperature=self.temperature,
                         base_temperature=self.base_temperature
                         )
        l2 = scl.forward(features=torch.Tensor(self.X_s.reshape(self.batch_size_s, 1, 128)),
                         labels=torch.Tensor(self.y_s)
                         )
        print('\nLosses from small batch size={}:'.format(self.batch_size_s))
        print('l1 = {}'.format(l1.numpy()))
        print('l2 = {}'.format(l2.numpy()))
        self.assertTrue(np.isfinite(l1.numpy()))
        self.assertTrue(np.isnan(l2.numpy()))


if __name__ == "__main__":
    unittest.main()
