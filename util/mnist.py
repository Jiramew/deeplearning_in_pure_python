import numpy as np
import struct


def load_mnist_images(images_path):
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(-1, 784)
    return images


def load_mnist_labels(labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    return labels


mnist_train_data = load_mnist_images("../data/train-images-idx3-ubyte")
mnist_train_label = load_mnist_labels("../data/train-labels-idx1-ubyte")

mnist_test_data = load_mnist_images("../data/t10k-images-idx3-ubyte")
mnist_test_label = load_mnist_labels("../data/t10k-labels-idx1-ubyte")


def image_to_binary(mat):
    return np.where(mat > 0, 1, 0)


def label_to_one_hot(label):
    label_set = [i for i in range(10)]
    num = len(label)
    one_hot_label = np.zeros([num, 10])
    offset = []
    index = []
    for i, c in enumerate(label):
        offset.append(i)
        index.append(label_set.index(c))
    one_hot_index = [offset, index]
    one_hot_label[one_hot_index] = 1.0
    return one_hot_label.astype(np.uint8)


def one_hot_to_label(label):
    label_set = [i for i in range(10)]
    label = np.array([[label_set[np.argmax(i)]] for i in label])
    return label
