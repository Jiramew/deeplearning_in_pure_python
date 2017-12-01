import numpy as np
from util.util import sigmoid_derivative, sigmoid, kl_divergence
import scipy.io
import scipy.optimize
from util import visual


class AutoencoderLinear(object):
    def __init__(self,
                 hidden_node_size,
                 train_data,
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

        self.sparsity = sparsity
        self.weight_decay = weight_decay
        self.penalty = penalty

        bound = np.sqrt(6) / np.sqrt(self.hidden_size + self.input_dim + 1)
        w1 = 2 * np.random.random((self.input_dim, self.hidden_size)) * bound - bound
        w2 = 2 * np.random.random((self.hidden_size, self.output_dim)) * bound - bound
        b1 = np.zeros((self.hidden_size, 1))
        b2 = np.zeros((self.output_dim, 1))

        self.theta = self.wrap_theta(w1, w2, b1, b2)

    @staticmethod
    def wrap_theta(w1, w2, b1, b2):
        return np.concatenate((w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()))

    def unwrap_theta(self, theta):
        return theta[0: self.hidden_size * self.input_dim].reshape((self.input_dim, self.hidden_size)), \
               theta[self.hidden_size * self.input_dim: 2 * self.hidden_size * self.input_dim].reshape(
                   (self.hidden_size, self.input_dim)), \
               theta[2 * self.hidden_size * self.input_dim: 2 * self.hidden_size * self.input_dim + self.hidden_size], \
               theta[2 * self.hidden_size * self.input_dim + self.hidden_size:]

    def cost(self, theta):
        input_data = self.train_data
        w1, w2, b1, b2 = self.unwrap_theta(theta)
        # hidden_z = np.dot(input_data, w1) + b1
        hidden_a = sigmoid(np.dot(input_data, w1) + b1)

        # output_z = np.dot(hidden_a, w2) + b2
        output_a = np.dot(hidden_a, w2) + b2  # output_z for linear decoder

        # average activation value rho_hat
        rho_hat = np.mean(hidden_a, axis=0)
        rho = self.sparsity

        diff = output_a - input_data
        sum_of_square_error = np.sum(np.multiply(diff, diff)) / (2 * self.input_num)
        weight_decay = self.weight_decay * (np.sum(np.multiply(w1, w1)) + np.sum(np.multiply(w2, w2))) / 2
        kl_diver = self.penalty * np.sum(kl_divergence(rho, rho_hat))

        # calculate the cost
        cost = sum_of_square_error + weight_decay + kl_diver

        # back propagation
        sparsity_delta = self.penalty * (-1.0 * (rho / rho_hat) + (1 - rho) / (1 - rho_hat))

        delta_output = np.mat(diff)  # * sigmoid_derivative(output_a))
        delta_hidden = np.multiply(np.dot(w2, delta_output.T) + np.transpose(np.mat(sparsity_delta)),
                                   sigmoid_derivative(hidden_a.T))

        grad = np.concatenate(((np.dot(input_data.T,
                                       delta_hidden.T) / self.input_num + self.weight_decay * w1).A.flatten(),
                               (np.dot(hidden_a.T, delta_output) / self.input_num + self.weight_decay * w2).A.flatten(),
                               (np.sum(delta_hidden, axis=1) / self.input_num).A.flatten(),
                               (np.sum(delta_output, axis=0) / self.input_num).A.flatten()))
        return [cost, grad]


if __name__ == '__main__':
    image_channels = 3
    patch_dim = 8
    n_patches = 100000

    input_dim = patch_dim * patch_dim * image_channels
    output_size = input_dim
    hidden_size = 400

    patches = scipy.io.loadmat("../data/stlSampledPatches")['patches']
    # show patches
    #  visual.displayColorNetwork(patches[:, 0:100])
    # normalize - zero mean
    patch_mean = np.mean(patches, axis=1)
    patches = patches - np.tile(patch_mean, (patches.shape[1], 1)).transpose()
    # ZCA whitening
    sigma = np.dot(patches, patches.transpose()) / patches.shape[1]
    (U, S, V) = np.linalg.svd(sigma)

    ZCA_white = np.dot(np.dot(U, np.diag(1 / np.sqrt(S + 0.1))), U.transpose())
    patch_ZCAwhite = np.dot(ZCA_white, patches)
    # show zcawhitened image
    #  visual.displayColorNetwork(patch_ZCAwhite[:, 0:100])
    # generate autoencoder
    encoder = AutoencoderLinear(hidden_node_size=hidden_size,
                                train_data=patch_ZCAwhite.T,
                                sparsity=0.035,
                                weight_decay=3e-3,
                                penalty=5)
    options = {'maxiter': 400, 'disp': True}
    J = lambda x: encoder.cost(x)
    # get w and b from BFGS
    result = scipy.optimize.minimize(J, encoder.theta, method='L-BFGS-B',
                                     jac=True, options=options)
    opt_theta = result.x

    W = opt_theta[0: hidden_size * input_dim].reshape(hidden_size, input_dim)
    b = opt_theta[2 * hidden_size * input_dim:2 * hidden_size * input_dim + hidden_size]
    visual.displayColorNetwork(W.dot(ZCA_white).transpose())
