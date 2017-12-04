import numpy as np
from sklearn import linear_model
from util.mnist import mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label

som = linear_model.LogisticRegression(solver="saga", multi_class="multinomial", verbose=1)

# data = np.array([[0, 0, 1],
#                  [0, 1, 1],
#                  [1, 0, 1],
#                  [1, 1, 1],
#                  [1, 0, 0],
#                  [1, 2, 1],
#                  [2, 3, 4],
#                  [0, 0, 1]])
#
# label = np.array([[0],
#                   [1],
#                   [1],
#                   [1],
#                   [0],
#                   [1],
#                   [1],
#                   [0]])
# som.fit(data,label)
# som.predict([[1, 0, 0]])

train_data = mnist_train_data
train_label = mnist_train_label
test_data = mnist_test_data
test_label = mnist_test_label

som.fit(train_data, train_label)

print(np.sum(som.predict(test_data) == test_label))
