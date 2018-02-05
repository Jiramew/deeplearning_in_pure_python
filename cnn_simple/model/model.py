from cnn_simple.util.constant import FORWARD_MODE_TRAINING, \
    BACKWARD_MODE_TRAINING, \
    LAYER_TYPE_INPUT_IMAGE

from cnn_simple.model.input_layer import InputLayer


class SimpleCnn(object):
    def __init__(self, mini_batch_size):
        self.layers = []
        self.next_layer = []

        self.next_layer_index = 0

        self.forward_mode = FORWARD_MODE_TRAINING
        self.backward_mode = BACKWARD_MODE_TRAINING

        self.learning_rate = 0.01
        self.momentum = 0.9
        self.l2 = 0

        self.mini_batch_size = mini_batch_size
        self.training_error = 0

        self.batch_learning_rate = None
        self.label_list_one_hot = []

    def add_layer(self):
        new_layer = InputLayer("image_input", 24, 24, 1, self)

        if self.next_layer_index == 0:
            if new_layer.layer_type != LAYER_TYPE_INPUT_IMAGE:
                raise Exception("First Layer should be input layer")
        else:
            pre_layer = self.layers[self.next_layer_index - 1]
            pre_layer.set_output_layer(new_layer)
            new_layer.set_input_layer(pre_layer)

        new_layer.layer_index = self.next_layer_index
        # self.layers[self.next_layer_index] = new_layer
        self.layers.append(new_layer)
        self.next_layer_index += 1

    def train(self, image_data_list, image_label_list):
        self.batch_learning_rate = self.learning_rate / self.mini_batch_size
        self.forward_mode = FORWARD_MODE_TRAINING
        self.backward_mode = BACKWARD_MODE_TRAINING

        self.training_error = 0

        self._one_hot(image_label_list)
        self._forward(image_data_list)

    def _one_hot(self, image_label_list):
        self.label_list_one_hot = [None] * self.mini_batch_size
        for i in range(0, self.mini_batch_size):
            self.label_list_one_hot[i] = [0] * self.layers[-1].units
            for j in range(0, self.layers[-1].units):
                self.label_list_one_hot[i][j] = 1 if j == image_label_list[i] else 0

    def _forward(self, image_data_list):
        self.layers[0].forward(image_data_list)
