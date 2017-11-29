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
