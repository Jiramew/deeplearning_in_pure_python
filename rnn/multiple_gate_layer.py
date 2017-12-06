import numpy as np


class MultipleGateLayer(object):
    def __init__(self):
        pass

    @staticmethod
    def forward(weight, input_data):
        return np.dot(weight, input_data)

    @staticmethod
    def backward(input_data, weight, loss):
        delta_weight = np.dot(loss.T, input_data)
        delta_data = np.dot(weight.T, loss)
        return delta_weight, delta_data
