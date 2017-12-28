import numpy as np


class SoftmaxLayer(object):
    def __init__(self):
        self.layer_type = "sm"
        self.output = None

    def forward(self, input_data):
        input_data -= np.max(input_data, axis=1).reshape([-1, 1])
        exp_sum = np.sum(np.exp(input_data), axis=1).reshape([-1, 1])
        self.output = np.exp(input_data) / exp_sum
        return self.output

    def backward(self, loss):
        loss = loss / self.output.shape[0]
        return loss, []
