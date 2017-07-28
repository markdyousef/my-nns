import unittest
import numpy as np
from numpy import random
from index import convolution, affine_forward

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

        # Test the affine_forward function
        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1,
                        0.5,
                        num=input_size
                        ).reshape(num_inputs, *input_shape)

        w = np.linspace(-0.2, 0.3,
                        num=weight_size
                        ).reshape(np.prod(input_shape), output_dim)

        b = np.linspace(-0.3, 0.1, num=output_dim)

        output, _ = affine_forward(x, w, b)

        correct_out = np.array([[1.49834967,  1.70660132,  1.91485297],
                                [3.25553199,  3.5141327,   3.77273342]])

        # Compare output with ours. The error should be around 1e-9.
        error = rel_error(output, correct_out)
        print('Testing affine_forward function:')
        print('difference: ', error)
        self.assertAlmostEquals(error, 0)


if __name__ == '__main__':
    unittest.main()
