from cnn_simple.model.layer import Layer
from cnn_simple.util.mat import Mat
from cnn_simple.util.mat_shape import MatShape
from cnn_simple.util.constant import INIT_ZEROS, LAYER_TYPE_INPUT_IMAGE


class InputLayer(Layer):
    def __init__(self, layer_name, image_width, image_height, image_depth, network):
        super(InputLayer, self).__init__(layer_name, 1)
        self.network = network
        self.layer_type = LAYER_TYPE_INPUT_IMAGE
        self.output_shape = MatShape(image_width, image_height, image_depth)
        self.output = []

    def forward(self, image_data_list):
        for i in range(0, self.network.mini_batch_size):
            self.output.append(Mat(self.output_shape, INIT_ZEROS))
            self.output[i].set_value_by_image(image_data_list[i], self.output_shape.depth)

    def backward(self):
        pass
