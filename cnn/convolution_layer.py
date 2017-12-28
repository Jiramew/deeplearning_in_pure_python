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
    def padding(input_data, kernel_size, stride, t="c"):
        batch_size, input_channel, input_row, input_col = input_data.shape
        row_length = math.ceil(input_row / stride[0])
        col_length = math.ceil(input_col / stride[1])
        pad = [kernel_size[0] + (row_length - 1) * stride[0] - input_row,
               kernel_size[1] + (col_length - 1) * stride[1] - input_col]

        pad_down = math.ceil(pad[0] / 2)
        pad_up = pad[0] - pad_down

        pad_right = math.ceil(pad[1] / 2)
        pad_left = pad[1] - pad_right

        pad_value = -1 * np.float32('inf') if t == 'p' else 0

        padded_output = np.pad(input_data,
                               ((0, 0),
                                (0, 0),
                                (pad_up, pad_down),
                                (pad_left, pad_right)),
                               'constant',
                               constant_values=pad_value)
        return row_length, col_length, padded_output

    @staticmethod
    def image_convolve(input_data, kernel_size, stride, row_length, col_length):
        kernel_row, kernel_col = kernel_size
        row_filter = [range(i * stride[0], i * stride[0] + kernel_row) for i in range(row_length)]
        col_filter = [range(i * stride[1], i * stride[1] + kernel_col) for i in range(col_length)]
        row = np.repeat(np.repeat(row_filter, kernel_col, axis=1), col_length, axis=0)
        col = np.tile(np.tile(col_filter, kernel_row), [row_length, 1])
        result = np.transpose(input_data[:, :, row, col],
                              [0, 2, 1, 3]).reshape(input_data.shape[0],
                                                    row_length * col_length,
                                                    -1)
        # get the [I_i,I_j] from i,j respectively from input_data with dim of I(or say J, they are in same dimension)
        return result

    @staticmethod
    def convolve_image(input_data, kernel_size, row_length, col_length):
        kernel_row, kernel_col = kernel_size
        row_filter = [range(i * row_length, (i + 1) * row_length) for i in range(row_length)]
        col_filter = [range(i * kernel_col, (i + 1) * kernel_col) for i in range(kernel_row)]
        row = np.repeat(np.repeat(row_filter, kernel_col, axis=1), kernel_row, axis=0)
        col = np.tile(np.tile(col_filter, col_length), [row_length, 1])
        return input_data[:, :, row, col]

    def padded(self, input_data):
        padded_shape = [self.batch_size,
                        self.output_channel,
                        self.kernel_row * 2 + input_data.shape[2] - 2,
                        self.kernel_col * 2 + input_data.shape[3] - 2]
        padded_loss = np.zeros(padded_shape)
        padded_loss[:, :, self.kernel_row - 1:self.kernel_row + input_data.shape[2] - 1,
        self.kernel_col - 1:self.kernel_col + input_data.shape[3] - 1] = input_data
        return padded_loss

    @staticmethod
    def reverse_padded(input_data, raw_row, raw_col):
        down = int(math.ceil((input_data.shape[2] - raw_row) / 2))
        up = int(input_data.shape[2] - raw_row - down)
        right = int(math.ceil((input_data.shape[3] - raw_col) / 2))
        left = int(input_data.shape[3] - raw_col - right)
        output = np.delete(input_data,
                           list(range(0, up)) + list(range(input_data.shape[2] - down,
                                                           input_data.shape[2])),
                           axis=2)
        output = np.delete(output,
                           list(range(0, left)) + list(range(input_data.shape[3] - right,
                                                             input_data.shape[3])),
                           axis=3)
        return output

    def rotate(self, input_data):
        rot_w = np.zeros_like(input_data)
        k = np.array([range(self.kernel_row - 1, -1, -1)]).T
        n = np.array([range(self.kernel_col - 1, -1, -1)])
        rot_w[:, :, k, n] = input_data
        return rot_w

    def forward(self, input_data):
        # convolve from padding and stride
        self.batch_size, self.input_channel, self.input_row, self.input_col = input_data.shape
        self.row_length, self.col_length, self.input_data = self.padding(input_data,
                                                                         [self.kernel_row, self.kernel_col],
                                                                         self.stride, "c")

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
        loss_row_length = loss.shape[2]
        loss_col_length = loss.shape[3]
        loss_padded_shape = [loss_row_length + (loss_row_length - 1) * (self.stride[0] - 1),
                             loss_col_length + (loss_col_length - 1) * (self.stride[1] - 1)]
        loss_padded_init = np.zeros(
            [self.batch_size,
             self.output_channel,
             loss_padded_shape[0],
             loss_padded_shape[1]])
        row = np.array([range(0, loss_padded_shape[0], self.stride[0])]).transpose()
        col = np.array([range(0, loss_padded_shape[1], self.stride[1])])
        loss_padded_init[:, :, row, col] = loss
        loss_padded = self.padded(loss_padded_init)

        grad_bias = np.sum(loss_padded_init, axis=(0, 2, 3))

        input_data_T = self.input_data.transpose([1, 0, 2, 3])
        grad_weight_col = self.image_convolve(input_data_T,
                                              loss_padded_init.shape[2:4],
                                              self.stride,
                                              self.kernel_row,
                                              self.kernel_col)
        loss_padded_init = np.transpose(loss_padded_init, [0, 2, 3, 1]).reshape(
            self.batch_size * loss_padded_shape[0] * loss_padded_shape[1], -1)
        grad_weight = np.dot(grad_weight_col, loss_padded_init)
        grad_weight = grad_weight.reshape([self.input_channel, self.kernel_row, self.kernel_col, self.output_channel])
        grad_weight = grad_weight.transpose(0, 3, 1, 2)

        rotation_weight = self.rotate(self.weight).transpose([1, 0, 2, 3])
        grad_col = self.image_convolve(loss_padded,
                                       rotation_weight.shape[2:4],
                                       self.stride,
                                       self.input_data.shape[2],
                                       self.input_data.shape[3])
        rotation_weight = np.transpose(rotation_weight, [0, 2, 3, 1]).reshape(
            [self.output_channel * self.kernel_row * self.kernel_col, -1])
        grad = np.dot(grad_col, rotation_weight)
        grad = grad.reshape([self.batch_size, self.input_data.shape[2], self.input_data.shape[3], self.input_channel])
        grad = grad.transpose(0, 3, 1, 2)

        grad = self.reverse_padded(grad, self.input_row, self.input_col)
        self.weight -= self.learning_rate * grad_weight
        self.bias -= self.learning_rate * grad_bias

        return grad, [grad_weight, grad_bias]


if __name__ == '__main__':
    from util.mnist import mnist_train_data, mnist_train_label


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
    cl = ConvolutionLayer([1, 32], [3, 3], [1, 1], 0.001)

    input_tensor = mnist_train_data[0:32].reshape(-1, 1, 28, 28)
    label_tensor = mnist_train_label[0:32]

    cl.forward(input_tensor)
