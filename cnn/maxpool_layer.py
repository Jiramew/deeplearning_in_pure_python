import numpy as np
from cnn.layer import Layer
from cnn.convolution_layer import ConvolutionLayer


class MaxpoolLayer(Layer):
    def __init__(self, kernel_size, stride):
        super(MaxpoolLayer).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.input_row = None
        self.input_col = None
        self.input_row_ori = None
        self.input_col_ori = None

        self.batch_size = None
        self.input_channel = None

        self.input_data = None
        self.row_length = None
        self.col_length = None

        self.index = None

    def forward(self, input_data):
        self.input_row_ori, self.input_col_ori = input_data.shape[2:4]
        self.batch_size, self.input_channel, self.input_row, self.input_col = input_data.shape
        self.row_length, self.col_length, self.input_data = ConvolutionLayer.padding(input_data,
                                                                                     self.kernel_size,
                                                                                     self.stride,
                                                                                     "p")
        self.output = ConvolutionLayer.image_convolve(self.input_data,
                                                      self.kernel_size,
                                                      self.stride,
                                                      self.row_length,
                                                      self.col_length)

        self.output = self.output.reshape(self.batch_size, self.row_length * self.col_length, self.input_channel, -1)
        self.output = self.output.transpose([0, 2, 1, 3])
        self.index = np.argmax(self.output, axis=3)
        self.output = np.max(self.output, axis=3).reshape(self.batch_size,
                                                          self.input_channel,
                                                          self.row_length,
                                                          self.col_length)
        # for each batch independently
        return self.output

    def backward(self, loss):
        loss = loss.reshape(self.batch_size, self.input_channel, self.row_length, self.col_length)
        grad = np.zeros([self.batch_size,
                         self.input_channel,
                         self.row_length * self.col_length,
                         self.kernel_size[0] * self.kernel_size[1]])
        i = np.repeat(np.tile(np.array([range(self.batch_size)]).T, [1, self.input_channel]),
                      self.row_length * self.col_length,
                      axis=1).reshape(self.index.shape)
        j = np.repeat(np.repeat([range(self.input_channel)], self.batch_size, axis=0),
                      self.row_length * self.col_length,
                      axis=1).reshape(self.index.shape)
        k = np.tile(range(self.row_length * self.col_length), self.batch_size * self.input_channel).reshape(
            self.index.shape)
        grad[i, j, k, self.index] = loss.reshape(self.batch_size, self.input_channel, -1)
        grad = ConvolutionLayer.convolve_image(grad, self.kernel_size, self.row_length, self.col_length)
        grad = ConvolutionLayer.reverse_padded(grad, self.input_row_ori, self.input_col_ori)
        return grad, []
