import json
import time
import numpy as np
from nn.full_connect_layer import FullConnectLayer
from nn.softmax_layer import SoftmaxLayer


class MultilayerNN(object):
    def __init__(self):
        self.layers = []
        self.grad = []
        self.label = None
        self.output = None

    def add(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input_data):
        data = input_data.copy()
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def back_propagation(self, output, label):
        loss = output - label
        for layer in reversed(self.layers):
            loss, grad = layer.backward(loss)
            self.grad.append(grad)

    def train(self, input_data, input_label):
        self.label = input_label
        self.output = self.forward_propagation(input_data)
        self.back_propagation(self.output, self.label)

    def predict(self, input_data):
        return self.forward_propagation(input_data)

    def loss(self):
        return -np.sum(self.label * np.log(self.output)) / self.label.shape[0]

    def accuracy(self):
        return np.mean(np.equal(np.argmax(self.output, axis=1), np.argmax(self.label, axis=1)))

    def save_model_as_json(self, prefix="model-"):
        model = {
            "fc_layers": [],
            "sm_layers": []
        }
        for layer in self.layers:
            if layer.layer_type == 'fc':
                fc_current_layer_info = {
                    'bias': layer.bias.tolist(),
                    'weight': layer.weight.tolist(),
                    'learning_rate': layer.learning_rate,
                    'batch_size': layer.batch_size,
                    'layer_type': layer.layer_type,
                }
                model["fc_layers"].append(fc_current_layer_info)
            if layer.layer_type == 'sm':
                sm_current_layer_info = {
                    'layer_type': layer.layer_type,
                }
                model["sm_layers"] = sm_current_layer_info
        model_json = json.dumps(model)
        with open('./' + prefix + str(int(time.time())) + '.model', 'a') as f:
            f.write(model_json)

        return model_json

    def load_model_from_json(self, file):
        with open(file, 'r') as f:
            model_json = json.loads(f.read())

        for fc in model_json['fc_layers']:
            bias = np.array(fc['bias'])
            weight = np.array(fc['weight'])
            learning_rate = fc['learning_rate']

            current_fc = FullConnectLayer(np.array(model_json['fc_layers'][0]['weight']).shape)
            current_fc.set_param(weight, bias, learning_rate)

            self.add(current_fc)
        for _ in model_json['sm_layers']:
            self.add(SoftmaxLayer())


def train():
    mnn = MultilayerNN()
    mnn.add(FullConnectLayer([784, 300], 0.003))
    mnn.add(FullConnectLayer([300, 100], 0.003))
    mnn.add(FullConnectLayer([100, 10], 0.003))
    mnn.add(SoftmaxLayer())

    from util.mnist import mnist_train_data, \
        mnist_train_label, image_to_binary, \
        label_to_one_hot, mnist_test_data, mnist_test_label

    test_data = image_to_binary(mnist_test_data)
    test_label = label_to_one_hot(mnist_test_label)

    batch_size = 32

    for i in range(1000):
        for j in range(60000 // batch_size):
            train_data = image_to_binary(mnist_train_data)
            train_label = label_to_one_hot(mnist_train_label)

            input_tensor = train_data[j * batch_size:(j + 1) * batch_size].reshape(-1, 1, 28, 28)
            label_tensor = train_label[j * batch_size:(j + 1) * batch_size]

            input_tensor = input_tensor.reshape([-1, 1, 28, 28])
            mnn.train(input_tensor, label_tensor)

            if j % 10 == 0:
                pred = mnn.predict(test_data)
                accuracy = np.mean(np.equal(np.argmax(pred, axis=1), np.argmax(test_label, axis=1)))
                print(mnn.loss(), mnn.accuracy(), accuracy)
            if j % 100 == 0:
                mnn.save_model_as_json()


def train_from_exist():
    mnn = MultilayerNN()
    mnn.load_model_from_json("./model-1514368830.model")

    from util.mnist import mnist_train_data, \
        mnist_train_label, image_to_binary, \
        label_to_one_hot, mnist_test_data, mnist_test_label

    test_data = image_to_binary(mnist_test_data)
    test_label = label_to_one_hot(mnist_test_label)

    batch_size = 32

    for i in range(1000):
        for j in range(60000 // batch_size):
            train_data = image_to_binary(mnist_train_data)
            train_label = label_to_one_hot(mnist_train_label)

            input_tensor = train_data[j * batch_size:(j + 1) * batch_size].reshape(-1, 1, 28, 28)
            label_tensor = train_label[j * batch_size:(j + 1) * batch_size]

            input_tensor = input_tensor.reshape([-1, 1, 28, 28])
            mnn.train(input_tensor, label_tensor)

            if j % 10 == 0:
                pred = mnn.predict(test_data)
                accuracy = np.mean(np.equal(np.argmax(pred, axis=1), np.argmax(test_label, axis=1)))
                print(mnn.loss(), mnn.accuracy(), accuracy)
            if j % 100 == 0:
                mnn.save_model_as_json()


def test():
    mnn = MultilayerNN()
    mnn.load_model_from_json("./model-1514368830.model")

    from util.mnist import mnist_test_data, mnist_test_label, image_to_binary, \
        label_to_one_hot

    test_data = image_to_binary(mnist_test_data)
    test_label = label_to_one_hot(mnist_test_label)

    pred = mnn.predict(test_data)
    accuracy = np.mean(np.equal(np.argmax(pred, axis=1), np.argmax(test_label, axis=1)))

    print(accuracy)


if __name__ == '__main__':
    train_from_exist()
    # train()
    # test()
