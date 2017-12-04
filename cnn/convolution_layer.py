import math
import numpy as np
from cnn.layer import Layer


class ConvolutionLayer(Layer):
    def __init__(self,
                 in_out_channel,
                 kernel_size,
                 stride=None,
                 lr=0.01):
        super(Layer).__init__()
        self.input_channel = in_out_channel[0]
        self.output_channel = in_out_channel[1]
        self.kernel_row = kernel_size[0]
        self.kernel_col = kernel_size[1]
        self.weight = np.random.randn(self.input_channel, self.output_channel, self.kernel_row, self.kernel_col)
        self.bias = np.random.random(self.output_channel)
        if stride is None:
            self.stride = [1, 1]
        else:
            self.stride = stride
        self.learning_rate = lr

        self.batch_size = None
        self.input_col = None
        self.input_row = None

        self.input_data = None
        self.row_length = None
        self.col_length = None

    @staticmethod
    def padding(input_data, kernel_size, stride):
        batch_size, input_channel, input_row, input_col = input_data.shape
        row_length = math.ceil(input_row / stride[0])
        col_index = math.ceil(input_col / stride[1])
        pad = [kernel_size[0] + (row_length - 1) * stride[0] - input_row,
               kernel_size[1] + (col_index - 1) * stride[1] - input_col]

        pad_down = math.ceil(pad[0] / 2)
        pad_up = pad[0] - pad_down
        pad_right = math.ceil(pad[1] / 2)
        pad_left = pad[1] - pad_right

        padded_output = np.pad(input_data,
                               ((0, 0),
                                (0, 0),
                                (pad_up, pad_down),
                                (pad_left, pad_right)),
                               'constant',
                               constant_values=0)
        return row_length, row_length, padded_output

    @staticmethod
    def image_convolve(input_x, ksize, stride, row_length, col_length):
        kernel_row, kernel_col = ksize
        i_fro = [range(i * stride[0], i * stride[0] + kernel_row) for i in range(row_length)]
        j_fro = [range(i * stride[1], i * stride[1] + kernel_col) for i in range(col_length)]
        i = np.repeat(np.repeat(i_fro, kernel_col, axis=1), col_length, axis=0)
        j = np.tile(np.tile(j_fro, kernel_row), [row_length, 1])
        x_col = np.transpose(input_x[:, :, i, j], [0, 2, 1, 3]).reshape(input_x.shape[0], row_length * col_length, -1)
        return x_col

    @staticmethod
    def convolve_image(input_x, ksize, row_length, col_length):
        kernel_row, kernel_col = ksize
        i_fro = [range(i * row_length, (i + 1) * row_length) for i in range(row_length)]
        j_fro = [range(i * kernel_col, (i + 1) * kernel_col) for i in range(kernel_row)]
        i = np.repeat(np.repeat(i_fro, kernel_col, axis=1), kernel_row, axis=0)
        j = np.tile(np.tile(j_fro, col_length), [row_length, 1])
        return input_x[:, :, i, j]

    def forward(self, input_data):
        # convolve from padding and stride
        self.batch_size, self.input_channel, self.input_row, self.input_col = input_data.shape
        self.row_length, self.col_length, self.input_data = self.padding(input_data,
                                                                         [self.kernel_row, self.kernel_col],
                                                                         self.stride)

        x_col = self.image_convolve(self.input_data,
                                    [self.kernel_row, self.kernel_col],
                                    self.stride,
                                    self.row_length,
                                    self.col_length)

        w = np.transpose(self.weight, [0, 2, 3, 1]).reshape(self.input_channel * self.kernel_row * self.kernel_col, -1)
        self.output = np.dot(x_col, w) + self.bias
        self.output = self.output.reshape(self.batch_size,
                                          self.row_length,
                                          self.col_length,
                                          self.output_channel).transpose(0, 3, 1, 2)
        return self.output

    def backward(self, loss):
        loss = loss.reshape((self.batch_size, self.output_channel, self.row_length, self.col_length))
        pass


if __name__ == '__main__':
    from util.mnist import mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label


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


    train_data = mnist_train_data
    train_label = label_to_one_hot(mnist_train_label)
    cl = ConvolutionLayer([1, 32], [5, 5], [1, 1], 0.001)

    input_tensor = mnist_train_data[0:32].reshape(-1, 1, 28, 28)
    label_tensor = mnist_train_label[0:32]

    cl.forward(input_tensor)
