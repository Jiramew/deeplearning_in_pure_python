import numpy as np


class Cnn(object):
    def __init__(self):
        self.cnn = []
        self.grad = []
        self.label = None
        self.output = None

    def add(self, layer):
        self.cnn.append(layer)

    def forward_propagation(self, input_data):
        data = input_data.copy()
        for layer in self.cnn:
            data = layer.forward(data)
        return data

    def back_propagation(self, output, label):
        loss = output - label
        for layer in reversed(self.cnn):
            error, grad = layer.backward(loss)
            self.grad.append(grad)

    def train(self, input_data, input_label):
        self.label = input_label
        self.output = self.forward_propagation(input_data)
        self.back_propagation(self.output, self.label)

    def loss(self):
        pass

    def accuracy(self):
        return np.mean(np.equal(np.argmax(self.output, axis=1), np.argmax(self.label, axis=1)))


if __name__ == '__main__':
    pass
