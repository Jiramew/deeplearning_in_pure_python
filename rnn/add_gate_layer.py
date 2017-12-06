import numpy as np


class AddGateLayer(object):
    def __init__(self):
        pass

    @staticmethod
    def forward(input_data1, input_data2):
        return input_data1 + input_data2

    @staticmethod
    def backward(input_data1, input_data2, loss):
        delta_data1 = loss * np.ones_like(input_data1)
        delta_data2 = loss * np.ones_like(input_data2)
        return delta_data1, delta_data2
