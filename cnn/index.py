import numpy as np
from numpy import random


def layer_forward(x, w):
    """ Receive inputs (x) and weights (w) """
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
    d_x = None
    d_w = None

    return d_x, d_w


def affine_forward(x, w, b):
    """ input:
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


def convolution(input, padding=0, filter_size=(2, 2),
                stride=2, output_depth=5):

    output_width = int((input.shape[0] - filter_size[0]
                        + 2*padding) / stride + 1)

    output_height = int((input.shape[1] - filter_size[1]
                        + 2*padding) / stride + 1)

    output = random.rand(output_width, output_height, output_depth)

    return output
