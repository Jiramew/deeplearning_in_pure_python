import numpy as np
from rnn.softmax_layer import SoftmaxLayer
from rnn.get_data import getSentenceData


class Rnn(object):  # recurrent_network
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 truncate=4,
                 lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.truncate = truncate
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
        T = len(input_data)
        state = np.zeros((T + 1, self.hidden_dim))
        state[-1] = np.zeros(self.hidden_dim)

        output = np.zeros((T, self.input_dim))

        for i in np.arange(T):
            sml = SoftmaxLayer()
            state[i] = np.tanh(self.U[:, input_data[i]] + self.W.dot(state[i - 1]))
            output[i] = sml.forward(self.V.dot(state[i]))

        return output, state

    def loss(self, input_data, input_label):
        loss = 0.0
        N = np.sum([len(y) for y in input_label])
        for i in np.arange(len(input_label)):
            output, state = self.forward(input_data[i])
            predict = output[np.arange(len(input_label[i])), input_label[i]]
            loss += -1 * np.sum(np.log(predict))

        return loss / N

    def bptt(self, input_data, input_label):
        T = len(input_label)

        output, state = self.forward(input_data)
        delta_u = np.zeros(self.U.shape)
        delta_w = np.zeros(self.W.shape)
        delta_v = np.zeros(self.V.shape)

        delta_output = output
        delta_output[np.arange(len(input_label))] -= 1

        for t in np.arange(T)[::-1]:
            delta_v += np.outer(delta_output[t], state[t].T)
            delta_t = np.dot(self.V.T, delta_output[t]) * (1 - (state[t] ** 2))
            for bptt_step in np.arange(max(0, t - self.truncate), t + 1)[::-1]:
                delta_w += np.outer(delta_t, state[bptt_step - 1])
                delta_u[:, input_data[bptt_step]] += delta_t
                delta_t = np.dot(self.W.T, delta_t) * (1 - state[bptt_step - 1] ** 2)
        return delta_u, delta_w, delta_v

    def predict(self, input_data):
        output, state = self.forward(input_data)
        return np.argmax(output, axis=1)

    def stochastic_gradient_decent(self, input_data, input_label, learning_rate):
        delta_u, delta_w, delta_v = self.bptt(input_data, input_label)
        self.U -= learning_rate * delta_u
        self.W -= learning_rate * delta_w
        self.V -= learning_rate * delta_v

    def train(self, input_data, input_label, learning_rate=0.005, epoch_num=100):
        loss_list = []
        for epoch in range(epoch_num):
            loss = self.loss(input_data, input_label)
            loss_list.append(loss)
            print(loss)

            for i in range(len(input_label)):
                self.stochastic_gradient_decent(input_data[i], input_label[i], learning_rate)

        return loss_list


if __name__ == '__main__':
    word_dim = 8000
    hidden_dim = 100
    X_train, Y_train = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

    np.random.seed(10)
    model = Rnn(word_dim, hidden_dim)

    o, s = model.forward(X_train[10])
    print(o.shape)
    print(o)

    predictions = model.predict(X_train[10])
    print(predictions.shape)
    print(predictions)

    print("Expected Loss for random predictions: %f" % np.log(word_dim))
    print("Actual loss: %f" % model.loss(X_train[:1000], Y_train[:1000]))

    loss = model.train(X_train[:100], Y_train[:100])
