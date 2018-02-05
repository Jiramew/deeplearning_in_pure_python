from cnn_simple.model.layer import Layer
from cnn_simple.util.mat import Mat
from cnn_simple.util.mat_shape import MatShape
from cnn_simple.util.constant import INIT_ZEROS, \
    INIT_XAVIER, \
    LAYER_TYPE_INPUT_IMAGE, \
    LAYER_TYPE_FULLY_CONNECTED, \
    ACTIVATION_TANH


class FCLayer(Layer):
    def __init__(self, name, units, activation):
        super(FCLayer, self).__init__(name, units)
        self.layer_type = LAYER_TYPE_FULLY_CONNECTED
        self.biases = [0] * units
        self.output_shape = MatShape(1, 1, units)
        self.activation = activation

        self.weight = []
        self.biases_grad = []
        self.weight_grad = []

    def set_input_layer(self, input_layer):
        super(FCLayer, self).set_input_layer(input_layer)
        for i in range(0, self.units):
            self.weight.append(Mat(self.input_shape, INIT_XAVIER))
            self.weight_grad.append(Mat(self.input_shape, INIT_ZEROS))

    def set_output_layer(self, output_layer):
        super(FCLayer, self).set_output_layer(output_layer)

    def set_params(self, weight, bias):
        for i in range(0, self.units):
            self.weight[i].set_value(weight[i])

        self.biases = bias
