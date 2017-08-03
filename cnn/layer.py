import numpy as np


def layer_forward(x, w):
    """
        input:
            - inputs (x): (N, d_1, ..., d_k),
            - weights (w): (D, M)
    """
    # intermediate value (z)
    z = None
    output = []
    cache = (x, w, z, output)

    return output, cache


def layer_backward(d_output, cache):
    """ Receive derivative of loss with respect
        to outputs and cache, and compute derivative
        with respect to inputs
    """

    # Unpack cache values
    x, w, z, output = cache

    # Compute derivatives (gradients)
    d_x, d_w = None, None

    return d_x, d_w


def affine_forward(x, w, b):
    """
        A simple linear feedforward (affine)
        input:
            - inputs (x): (N, d_1, ..., d_k),
            - weights (w): (D, M)
            - bias (b): (M,)
        return:
            - output: (N, M)
            - cache: (x, w, b)
    """
    N = x.shape[0]

    # reshape input into rows
    output = x.reshape([N, -1]).dot(w) + b
    cache = (x, w, b)

    return output, cache


def affine_backward(d_output, cache):
    """
        input:
            - upstream derivative (d_output): (N, M)
            - cache (cache): (x, w)
        return:
            - gradients (dx, d_w, d_b): ((N, d1, ..., d_k)(D, M), (M,))
    """

    # Unpack cache values
    x, w, b = cache

    N = d_output.shape[0]
    d_x = d_output.dot(w.T).reshape(x.shape)
    d_w = x.reshape([N, -1]).T.dot(d_output)
    d_b = np.sum(d_output, axis=0)

    return d_x, d_w, d_b


def relu_forward(x):
    """
        input:
            - inputs (x): (N, d_1, ..., d_k)
        return:
            - output: (N, d_1, ..., d_k)
            - cache: x
    """
    output = np.fmax(x, 0)
    cache = x

    return output, cache


def relu_backward(d_output, cache):
    """
        input:
            - upstream derivative (d_output): (N, d_1, ..., d_k)
            - cache for x (cache): (N, d_1, ..., d_k)
        return:
            - d_x: gradient with respect to x
    """
    x = cache
    d_x = np.sign(np.fmax(x, 0)) * d_output

    return d_x
