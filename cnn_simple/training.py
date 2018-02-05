import cv2
import random
from cnn_simple.util.constant import \
    LAYER_TYPE_CONV, \
    LAYER_TYPE_MAX_POOL, \
    LAYER_TYPE_INPUT_IMAGE, \
    LAYER_TYPE_FULLY_CONNECTED, \
    ACTIVATION_SOFTMAX
from cnn_simple.model.model import SimpleCnn
from cnn_simple.util.image_data import ImageData


class Training(object):
    def __init__(self):
        self.train_num = 50000
        self.test_num = 10000
        self.validate_num = 10000

        self.test_offset = 60000
        self.validate_offset = 50000

        self.image_per_file_num = 10000

        self.mini_batch_size = 20
        self.validate_size = 100

        self.iter = 0
        self.epoch = 0
        self.example_seen = 0

        self.train_images = []
        self.test_images = []
        self.validate_images = []
        self.labels = None

        self.model = None

        self.load_data_from_file()
        self.initialize_network()

    def load_data_from_file(self):
        train_file_template = "./resource/mnist_training_{0}.png"
        for i in range(0, 5):
            self.train_images.append(cv2.cvtColor(cv2.imread(train_file_template.format(i)), cv2.COLOR_BGR2RGBA))

        test_file_template = "./resource/mnist_test.png"
        self.test_images.append(cv2.cvtColor(cv2.imread(test_file_template), cv2.COLOR_BGR2RGBA))

        validate_file_template = "./resource/mnist_validation.png"
        self.validate_images.append(cv2.cvtColor(cv2.imread(validate_file_template), cv2.COLOR_BGR2RGBA))

        label_file_template = "./resource/mnist_label.txt"
        with open(label_file_template, 'r') as f:
            self.labels = [int(s) for s in f.read().strip().split(",")]

    def initialize_network(self):
        self.model = SimpleCnn(self.mini_batch_size)
        self.model.add_layer()

    def train(self):
        if self.model is None:
            return

        if self.iter < self.train_num:
            image_file_index = self.iter // self.image_per_file_num

            train_image_batch = []
            train_label_batch = []

            i = 0
            while True:
                if i < self.mini_batch_size and self.iter < self.train_num:
                    train_image_batch.append(self.get_train_image_data(self.iter))
                    train_label_batch.append(self.labels[self.iter])
                    i += 1
                    self.iter += 1
                else:
                    break
            self.example_seen += self.mini_batch_size
            self.model.train(train_image_batch, train_label_batch)

            self.epoch += 1

    def get_train_image_data(self, ite):
        image_file_index = ite // self.image_per_file_num
        image_index = ite % self.image_per_file_num

        start_x = (image_index % 100) + random.randint(0, 4)
        start_y = (ite // 100) + random.randint(0, 4)

        return ImageData(self.train_images[image_file_index][start_y:(start_y + 24), start_x:(start_x + 24)])


if __name__ == '__main__':
    tr = Training()
    tr.load_data_from_file()
    tr.train()
