import unittest
import numpy as np
from numpy import random
from index import convolution

# Hyperparameters
P = 0  # zero padding
F = 5  # filter size
S = 2  # stride
K = 5  # depth

# Input volume
X = random.rand(11, 11, 4)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class MyTest(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(
            convolution(X,
                        padding=P,
                        filter_size=(F, F),
                        stride=S,
                        output_depth=K
                        ).shape,
            (4, 4, 5)
        )

if __name__ == '__main__':
    unittest.main()
