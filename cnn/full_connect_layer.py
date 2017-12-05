import numpy as np
from cnn.layer import Layer


class FullConnectLayer(Layer):
    def __init__(self, weight, lr=0.01):
        super(FullConnectLayer).__init__()
        self.weight = np.random.randn(weight[0], weight[1]) / np.sqrt(weight[0])
        self.bias = np.random.random((1, weight[1]))
        self.learning_rate = lr
        self.batch_size = None
        self.input_data = None

    def forward(self, input_data):
        self.batch_size = input_data.shape[0]
        self.input_data = np.reshape(input_data, [self.batch_size, -1])
        self.output = np.dot(self.input_data, self.weight) + self.bias
        return self.output

    def backward(self, loss):
        weight_grad = np.dot(self.input_data.T, loss)
        bias_grad = np.sum(loss, axis=0).reshape(1, -1)
        loss = np.dot(loss, self.weight.T)
        self.weight -= weight_grad * self.learning_rate
        self.bias -= bias_grad * self.learning_rate
        return loss, [weight_grad, bias_grad]
