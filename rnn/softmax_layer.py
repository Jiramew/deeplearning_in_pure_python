import numpy as np


class SoftmaxLayer:
    def forward(self, x):
        exp_scores = np.exp(x - np.max(x))
        return exp_scores / np.sum(exp_scores)
