from sklearn.neighbors import NearestNeighbors

from util.mnist import mnist_train_data, \
    mnist_train_label, mnist_test_data, mnist_test_label, image_to_binary, \
    label_to_one_hot, one_hot_to_label

nn = NearestNeighbors(n_neighbors=3, metric="l2", n_jobs=-1, algorithm='kd_tree')

train_data = image_to_binary(mnist_train_data)
train_label = label_to_one_hot(mnist_train_label)
test_data = image_to_binary(mnist_test_data)
test_label = mnist_test_label

nn.fit(train_data)

print(nn.kneighbors(test_data))
