import numpy as np


class Softmax(object):
    def __init__(self,
                 train_data,
                 train_label,
                 lr=0.00001,
                 wd=0.1,
                 turn=100):
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
        return (np.exp(np.dot(self.train_data, self.theta)).T / np.sum(
            np.exp(np.dot(self.train_data, self.theta)), axis=1)).T

    def theta_init(self):
        return np.zeros((self.input_dim, self.output_dim))

    def train(self):
        for i in range(self.turn):
            print('loop %d' % i)
            output = self.output()
            loss = self.train_label - output
            self.theta += self.learning_rate * (
                np.dot(self.train_data.T, loss) / self.input_num - self.weight_decay * self.theta)

            # if (i % 1000) == 0:
            print(np.mean(np.abs(loss)))

    def predict(self, test_data):
        test_data = np.insert(test_data, test_data.shape[1], np.ones(test_data.shape[0]), 1)
        return (np.exp(np.dot(test_data, self.theta)).T / np.sum(np.exp(np.dot(test_data, self.theta)), axis=1)).T


def mnist_test():
    from util.mnist import mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label

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

    train_data = mnist_train_data
    train_label = label_to_one_hot(mnist_train_label)
    test_data = mnist_test_data
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
