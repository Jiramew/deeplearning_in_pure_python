import numpy as np
from rnn.recurrent_layer import RecurrentLayer
from rnn.softmax_layer import SoftmaxLayer


class Rnn(object):  # recurrent_network
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state = [np.zeros((self.hidden_dim, 1))]
        self.U = np.random.uniform(-np.sqrt(1.0 / self.input_dim),
                                   np.sqrt(1.0 / self.input_dim),
                                   (self.hidden_dim, self.input_dim))  # with input
        self.W = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim),
                                   np.sqrt(1.0 / self.hidden_dim),
                                   (self.hidden_dim, self.hidden_dim))  # with state(t-1)
        self.V = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim),
                                   np.sqrt(1.0 / self.hidden_dim),
                                   (self.input_dim, self.hidden_dim))  # with state(t)

        self.learning_rate = lr

    def forward(self, input_data):
        state_previous = np.zeros((self.hidden_dim, 1))
        layer_list = []
        for row in input_data:
            rnn_layer = RecurrentLayer()
            input_vector = np.zeros(self.input_dim)
            input_vector[row] = 1
            rnn_layer.forward(input_vector, state_previous, self.U, self.W, self.V)
            state_previous = rnn_layer.state
            layer_list.append(rnn_layer)
        return layer_list

    def loss(self, input_data, input_label):
        loss = 0.0
        for i in range(len(input_label)):
            sml = SoftmaxLayer()
            layers = self.forward(input_data[i])
            tmp_loss = 0.0
            for j, layer in enumerate(layers):
                tmp_loss += sml.loss(layer.mul_state_v, input_label[j])
            loss += tmp_loss / len(input_label[i])

        return loss / len(input_label)

    def bptt(self, input_data, input_label):
        pass

    def predict(self, input_data):
        sml = SoftmaxLayer()
        layers = self.forward(input_data)
        return [np.argmax(sml.forward(layer.mul_state_v)) for layer in layers]


if __name__ == '__main__':
    '''
    chars 'h','e','l','o',is represented as follows
    'h' <---[1,0,0,0]
    'e' <---[0,1,0,0]
    'l' <---[0,0,1,0]
    'o' <---[0,0,0,1]
    '''
x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
y = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

rnn = Rnn(4, 4, 4)
rnn.forward(x)
