import numpy as np


class LinearRegression(object):
    def __init__(self,
                 train_data,
                 train_label,
                 lr=0.001,
                 turn=50000):
        self.train_data = train_data
        self.train_label = train_label
        self.input_dim = self.train_data.shape[
            1]  # number of nodes in input unit(train data) not including bias node aka. the dimension of train data.
        self.input_num = self.train_data.shape[0]  # number of instance
        self.output_dim = self.train_label.shape[1]

        self.learning_rate = lr
        self.turn = turn
        self.theta, self.b = self.param_init()

    def param_init(self):
        return np.random.randn(1, self.input_dim), np.random.randn(1, self.output_dim)

    def train(self):
        for i in range(self.turn):
            output = np.dot(self.train_data, self.theta.T) + np.tile(self.b, (self.input_num, 1))
            loss = output - self.train_label
            self.theta -= self.learning_rate * np.dot(loss.T, self.train_data).reshape(-1) / self.input_num
            self.b -= self.learning_rate * np.mean(loss.T.reshape(-1))

            if (i % 1000) == 0:
                self.check_params()
                print(np.mean(np.abs(loss)))

    def check_params(self):
        print(self.theta, self.b)


if __name__ == '__main__':
    data = np.array([[0, 0, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 1],
                     [1, 2, 1],
                     [2, 3, 4],
                     [0, 0, 1],
                     [1, 2, 3],
                     [1, 2, 4]])

    label = np.array([[0],
                      [2],
                      [2],
                      [3],
                      [1],
                      [4],
                      [9],
                      [1],
                      [6],
                      [7]])
    lir = LinearRegression(data, label)
    lir.train()
