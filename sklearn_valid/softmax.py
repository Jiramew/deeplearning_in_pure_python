import numpy as np
from sklearn import linear_model
from util.mnist import mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label

som = linear_model.LogisticRegression(solver="saga", multi_class="multinomial", verbose=1)

train_data = mnist_train_data
train_label = mnist_train_label
test_data = mnist_test_data
test_label = mnist_test_label

som.fit(train_data, train_label)

print(np.sum(som.predict(test_data) == test_label))
