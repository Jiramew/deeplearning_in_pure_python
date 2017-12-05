import numpy as np
from cnn.full_connect_layer import FullConnectLayer
from cnn.maxpool_layer import MaxpoolLayer
from cnn.convolution_layer import ConvolutionLayer
from cnn.relu_layer import ReluLayer
from cnn.softmax_layer import SoftmaxLayer


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
            loss, grad = layer.backward(loss)
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
    cnn = Cnn()
    cnn.add(ConvolutionLayer([1, 32], [5, 5], [1, 1], 0.001))
    cnn.add(MaxpoolLayer([2, 2], [2, 2]))
    cnn.add(ConvolutionLayer([32, 64], [5, 5], [1, 1], 0.001))
    cnn.add(MaxpoolLayer([2, 2], [2, 2]))
    cnn.add(FullConnectLayer([7 * 7 * 64, 1024], 0.001))
    cnn.add(ReluLayer())
    cnn.add(FullConnectLayer([1024, 10], 0.001))
    cnn.add(SoftmaxLayer())

    from util.mnist import mnist_train_data, mnist_train_label


    def image_to_binary(mat):
        return np.where(mat > 0, 1, 0)


    def label_to_one_hot(label):
        label_set = [i for i in range(10)]
        num = len(label)
        one_hot_label = np.zeros([num, 10])
        offset = []
        index = []
        for i, c in enumerate(label):
            offset.append(i)
            index.append(label_set.index(c))
        one_hot_index = [offset, index]
        one_hot_label[one_hot_index] = 1.0
        return one_hot_label.astype(np.uint8)


    def one_hot_to_label(label):
        label_set = [i for i in range(10)]
        label = np.array([[label_set[np.argmax(i)]] for i in label])
        return label


    for i in range(10):
        for j in range(60000 // 32):
            train_data = image_to_binary(mnist_train_data)
            train_label = label_to_one_hot(mnist_train_label)

            input_tensor = train_data[j * 32:(j + 1) * 32].reshape(-1, 1, 28, 28)
            label_tensor = train_label[j * 32:(j + 1) * 32]

            input_tensor = input_tensor.reshape([-1, 1, 28, 28])
            cnn.train(input_tensor, label_tensor)

            if j % 10 == 0:
                print(cnn.loss(), cnn.accuracy())
