import numpy as np
from numpy import random


def convolution(input, padding=0, filter_size=(2, 2),
                stride=2, output_depth=5):

    output_width = int((input.shape[0] - filter_size[0]
                        + 2*padding) / stride + 1)

    output_height = int((input.shape[1] - filter_size[1]
                        + 2*padding) / stride + 1)

    output = random.rand(output_width, output_height, output_depth)

    return output
