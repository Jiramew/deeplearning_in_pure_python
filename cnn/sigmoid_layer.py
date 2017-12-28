import numpy as np
from cnn.layer import Layer


class SigmoidLayer(Layer):
    def __init__(self):
        super(SigmoidLayer).__init__()

    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-1.0 * input_data))
        return self.output

    def backward(self, loss):
        loss = (loss * self.output * (1 - self.output)) / loss.shape[0]
        return loss, []
