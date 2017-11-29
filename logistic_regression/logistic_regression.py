import numpy as np
from util.util import sigmoid, sigmoid_derivative


class LogisticRegression(object):
    def __init__(self,
                 train_data,
                 train_label,
                 lr=0.1,
                 rg=1,
                 th=0.5,
                 turn=10000):
        self.train_data = train_data
        self.train_label = train_label
        self.input_dim = self.train_data.shape[
            1]  # number of nodes in input unit(train data) not including bias node aka. the dimension of train data.
        self.input_num = self.train_data.shape[0]  # number of instance

        self.learning_rate = lr
        self.regularization = rg
        self.threshold = th
        self.turn = turn

        self.theta = self.theta_init()

    def theta_init(self):
        return np.zeros(self.input_dim)

    def train(self):
        for i in range(self.turn):
            output = sigmoid(np.dot(self.train_data, self.theta))
            loss = self.train_label.T - output
            self.theta += self.learning_rate * np.dot(loss, self.train_data)[0]

            if (i % 1000) == 0:
                print(np.mean(np.abs(loss)))

    def predict(self, test_data):
        return 1 if sigmoid(np.dot(test_data, self.theta)) > self.threshold else 0


if __name__ == '__main__':
    data = np.array([[0, 0, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 1],
                     [1, 2, 1],
                     [2, 3, 4],
                     [0, 0, 1]])

    label = np.array([[0],
                      [1],
                      [1],
                      [1],
                      [0],
                      [1],
                      [1],
                      [0]])

    lr = LogisticRegression(data, label)
    lr.train()

    test = np.array([[5, 3, 1]])
    print(lr.predict(test))
    a = 1
