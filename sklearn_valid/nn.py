from sklearn import neural_network
import numpy as np

nn = neural_network.MLPClassifier(
    hidden_layer_sizes=(6, 6),
    activation="logistic",
    solver="sgd",
    alpha=0.1,
    learning_rate="constant",
    learning_rate_init=0.2,
    max_iter=10000
)
np.random.seed(0)
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
                  [0]]).reshape(-1)

nn.fit(data, label)

test = np.array([[0, 0, 1]])
print(nn.n_iter_)
print(nn.coefs_)
print(nn.intercepts_)
print(nn.predict(test))  # needs to print out the activation value
