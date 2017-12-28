import numpy as np
from util.util import sigmoid_derivative, sigmoid, kl_divergence


class Autoencoder(object):
    def __init__(self,
                 hidden_node_size,
                 train_data,
                 turn=10,
                 sparsity=0.1,
                 weight_decay=3e-3,
                 penalty=3):
        np.random.seed(0)
        self.train_data = train_data
        self.hidden_size = hidden_node_size  # number nodes in the hidden layer
        self.all_layer_num = 3
        self.input_dim = self.train_data.shape[
            1]  # number of nodes in input unit(train data) not including bias node aka. the dimension of train data.
        self.input_num = self.train_data.shape[0]  # number of instance
        self.output_dim = self.input_dim  # output dim

        self.turn = turn
        self.sparsity = sparsity
        self.weight_decay = weight_decay
        self.penalty = penalty

        self.weight_list = self.init_weight()  # init the weight
        self.bias_list = self.init_bias()
        self.weight_gradient_list = []
        self.bias_gradient_list = []
        self.theta = self.set_params_theta()

    def init_weight(self):
        weight = []
        bound = np.sqrt(6) / np.sqrt(self.hidden_size + self.input_dim + 1)
        weight_input_to_hidden_first = 2 * np.random.random((self.input_dim, self.hidden_size)) * bound - bound
        weight.append(weight_input_to_hidden_first)

        weight_hidden_last_to_output = 2 * np.random.random((self.hidden_size, self.output_dim)) * bound - bound
        weight.append(weight_hidden_last_to_output)

        return weight

    def init_bias(self):
        bias = [np.zeros(self.hidden_size),
                np.zeros(self.output_dim)]
        return bias

    def set_params_theta(self):
        return np.concatenate(
            (self.weight_list[0].flatten(), self.weight_list[1].flatten(), self.bias_list[0], self.bias_list[1]))

    def cost(self):
        # forward propagation
        hidden_z = np.dot(self.train_data, self.weight_list[0]) + np.tile(self.bias_list[0], (self.input_num, 1))
        hidden_a = sigmoid(hidden_z)

        output_z = np.dot(hidden_a, self.weight_list[1]) + np.tile(self.bias_list[1], (self.input_num, 1))
        output_a = sigmoid(output_z)

        # average activation value rho_hat
        rho_hat = np.mean(hidden_a, axis=0)
        rho = np.tile(self.sparsity, self.hidden_size)

        # calculate the cost
        cost = np.sum((self.train_data - output_a) ** 2) / (2 * self.input_num) + \
               self.weight_decay * (np.sum(self.weight_list[0] ** 2) + np.sum(self.weight_list[1] ** 2)) / 2 + \
               self.penalty * np.sum(kl_divergence(rho, rho_hat))

        # back propagation
        sparsity_delta = np.tile(-1.0 * (rho / rho_hat) + (1 - rho) / (1 - rho_hat), (self.input_num, 1)).T

        delta_output = -1.0 * (self.train_data - output_a) * sigmoid_derivative(output_a)
        delta_hidden = (np.dot(self.weight_list[1],
                               delta_output.T) + self.penalty * sparsity_delta) * sigmoid_derivative(hidden_a.T)

        # param gradient
        self.weight_gradient_list = [
            np.dot(self.train_data.T, delta_hidden.T) / self.input_num + self.weight_decay * self.weight_list[0],
            np.dot(hidden_a.T, delta_output) / self.input_num + self.weight_decay * self.weight_list[1]
        ]
        self.bias_gradient_list = [np.sum(delta_hidden, axis=1) / self.input_num,
                                   np.sum(delta_output, axis=1) / self.input_num]

        grad = np.concatenate((self.weight_gradient_list[0].flatten(),
                               self.weight_gradient_list[1].flatten(),
                               self.bias_gradient_list[0].flatten(),
                               self.bias_gradient_list[1].flatten()))
        return grad, cost


if __name__ == '__main__':
    data = np.array([[0, 0, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 1],
                     [1, 2, 1],
                     [2, 3, 4],
                     [0, 0, 1]])

    ae = Autoencoder(5, data)
    ae.cost()

    a = 1
