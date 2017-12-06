import numpy as np


class Softmax(object):
    def __init__(self,
                 train_data,
                 train_label,
                 lr=0.25,
                 wd=0.01,
                 turn=5000):
        self.train_data = np.insert(train_data, 0, np.ones(train_data.shape[0]), 1)
        self.train_label = train_label
        self.input_dim = self.train_data.shape[
            1]  # number of nodes in input unit(train data) not including bias node aka. the dimension of train data.
        self.input_num = self.train_data.shape[0]  # number of instance
        self.output_dim = self.train_label.shape[1]

        self.learning_rate = lr
        self.weight_decay = wd
        self.theta = self.theta_init()
        self.turn = turn

    def output(self):
        value = np.dot(self.train_data, self.theta)
        value -= np.max(value)
        return (np.exp(value).T / np.sum(
            np.exp(value), axis=1)).T

    def theta_init(self):
        return np.ones((self.input_dim, self.output_dim))

    def train(self):
        for i in range(self.turn):
            print('loop %d' % i)
            # self.learning_rate += (i / 3000) * (0.1 ** (i // 100 + 1)) if i < 200 else (i / 300) * (
            # 0.1 ** (i // 100 + 1))
            output = self.output()
            loss = self.train_label - output
            self.theta += self.learning_rate * (
                np.dot(self.train_data.T, loss) / self.input_num - self.weight_decay * self.theta)

            # if (i % 1000) == 0:
            print(np.mean(np.abs(loss)), self.learning_rate)

    def predict(self, test_data):
        test_data = np.insert(test_data, 0, np.ones(test_data.shape[0]), 1)
        value = np.dot(test_data, self.theta)
        value -= np.max(value)
        return (np.exp(value).T / np.sum(np.exp(value), axis=1)).T


def mnist_test():
    from util.mnist import mnist_train_data, \
        mnist_train_label, mnist_test_data, mnist_test_label, image_to_binary, \
        label_to_one_hot, one_hot_to_label

    train_data = image_to_binary(mnist_train_data)
    train_label = label_to_one_hot(mnist_train_label)
    test_data = image_to_binary(mnist_test_data)
    test_label = mnist_test_label

    sm = Softmax(train_data, train_label)
    sm.train()

    pred = one_hot_to_label(sm.predict(test_data)).reshape(-1)
    print(np.sum(pred == test_label))


if __name__ == '__main__':
    mnist_test()
    # data = np.array([[0, 0, 1],
    #                  [0, 1, 1],
    #                  [1, 0, 1],
    #                  [1, 1, 1],
    #                  [1, 0, 0],
    #                  [1, 2, 1],
    #                  [2, 3, 4],
    #                  [0, 0, 1]])
    #
    # label = np.array([[0, 1],
    #                   [1, 0],
    #                   [1, 0],
    #                   [1, 0],
    #                   [0, 1],
    #                   [1, 0],
    #                   [1, 0],
    #                   [0, 1]])
    #
    # sm = Softmax(data, label)
    # sm.train()
    #
    # test = np.array([[1, 0, 0]])
    # print(sm.predict(test))
