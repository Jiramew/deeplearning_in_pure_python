import numpy as np


def sigmoid(x):
    return np.longfloat(1.0 / (1 + np.exp(-x)))


def sigmoid_derivative(value):
    return np.longfloat(value * (1 - value))


def tanh_derivative(value):
    return np.longfloat(1.0 - value ** 2)


def relu(x):
    res = np.maximum(0, x)
    return res


def uniform_random_array(a, b, *arg):
    np.random.seed(0)
    return np.random.rand(*arg) * (b - a) + a


def kl_divergence(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
