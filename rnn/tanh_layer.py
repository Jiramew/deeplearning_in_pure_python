import numpy as np


class TanhLayer(object):
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data):
        return np.tanh(input_data)

    @staticmethod
    def backward(input_data, loss):
        output = self.forward(input_data)
        return (1.0 - np.square(output)) * loss
