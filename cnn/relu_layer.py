import numpy as np
from cnn.layer import Layer


class ReluLayer(Layer):
    def __init__(self):
        super(ReluLayer).__init__()

    def forward(self, input_data):
        self.output = input_data
        self.output = np.maximum(self.output, 0)
        return self.output

    def backward(self, loss):
        loss = loss.reshape(self.output.shape[0], -1)
        loss = np.where(self.output > 0, loss, 0)
        return loss, []
