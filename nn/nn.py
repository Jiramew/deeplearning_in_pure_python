import numpy as np
from util.util import sigmoid, sigmoid_derivative

np.random.seed(0)
data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [1, 2, 1]])
label = np.array([[0],
                  [1],
                  [1],
                  [1],
                  [0],
                  [1]])

num_example = len(data)

alpha = 1
input_dim = 3
hidden_dim = 4
output_dim = 1

lr = 0.1

w_ih = 2 * np.random.random((input_dim, hidden_dim)) - 1
w_ho = 2 * np.random.random((hidden_dim, output_dim)) - 1

w_ih_update = np.zeros_like(w_ih)
w_ho_update = np.zeros_like(w_ho)

for i in range(10000):
    X = data
    y = label

    # forward propagation
    hidden_layer_input = np.dot(X, w_ih)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, w_ho)
    output_layer_output = sigmoid(output_layer_input)

    # back propagation
    loss_output_layer = y - output_layer_output
    delta_output_layer = - loss_output_layer * sigmoid_derivative(output_layer_output)

    loss_hidden_layer = np.dot(delta_output_layer, w_ho.T)
    delta_hidden_layer = loss_hidden_layer * sigmoid_derivative(hidden_layer_output)

    if (i % 1000) == 0:
        print(np.mean(np.abs(loss_output_layer)))

    w_ih_update = np.dot(X.T, delta_hidden_layer)
    w_ho_update = np.dot(hidden_layer_output.T, delta_output_layer)

    w_ih -= lr * w_ih_update
    w_ho -= lr * w_ho_update
