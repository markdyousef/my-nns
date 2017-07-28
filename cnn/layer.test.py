import unittest
import numpy as np
from layer import affine_forward, affine_backward


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)

    # iterator
    it = np.nditer(x,
                   flags=['multi_index'],
                   op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index

        old_val = x[ix]
        x[ix] = old_val + h

        pos = f(x).copy()
        x[ix] = old_val - h

        neg = f(x).copy()
        x[ix] = old_val

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()

    return grad


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class LayerTest(unittest.TestCase):
    def test(self):
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
        # print('difference: ', error)
        self.assertAlmostEqual(error, 0)

        # Test the affine_backward function
        np.random.seed(231)
        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dx_num = eval_numerical_gradient_array(
            lambda x: affine_forward(x, w, b)[0], x, dout)

        dw_num = eval_numerical_gradient_array(
            lambda w: affine_forward(x, w, b)[0], w, dout)
        db_num = eval_numerical_gradient_array(
            lambda b: affine_forward(x, w, b)[0], b, dout)

        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)

        # The error should be around 1e-10
        print('Testing affine_backward function:')
        # print('dx error: ', rel_error(dx_num, dx))
        # print('dw error: ', rel_error(dw_num, dw))
        # print('db error: ', rel_error(db_num, db))
        self.assertAlmostEqual(rel_error(dx_num, dx), 1e-10)
        self.assertAlmostEqual(rel_error(dw_num, dw), 1e-10)
        self.assertAlmostEqual(rel_error(db_num, db), 1e-10)


if __name__ == '__main__':
    unittest.main()
