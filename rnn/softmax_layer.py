import numpy as np


class SoftmaxLayer:
    def forward(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self, x, y):  # y^-y
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs
