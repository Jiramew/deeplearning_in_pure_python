import numpy as np
from cnn.layer import Layer


class ReluLayer(Layer):
    def __init__(self):
        super(ReluLayer).__init__()

    def forward(self, input_data):
        self.output = input_data
        return np.maximum(self.output, 0)

    def backward(self, loss):
        loss = loss.reshape(self.output.shape)
        loss = np.where(self.output > 0, loss, 0)
        return loss, []
