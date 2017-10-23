__author__ = 'raphey'

import numpy as np


class Network(object):
    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, x_in):
        self.layers[0].input = x_in
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].input = self.layers[i].forward_pass()
        self.layers[-1].forward_pass()
        return self.layers[-1].output

    def feed_backward(self, delta_y_out, alpha):
        self.layers[-1].output_side_deltas = delta_y_out
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i - 1].output_side_deltas = self.layers[i].backward_pass(alpha)
        self.layers[0].backward_pass(alpha)

    def train(self, x_data, y_data, alpha, epochs, verbose=False):
        x_training, x_testing = x_data['train'], x_data['test']
        y_training, y_testing = y_data['train'], y_data['test']
        for e in range(1, epochs + 1):
            delta_y = y_training - self.feed_forward(x_training)
            self.feed_backward(delta_y, alpha)


class Layer(object):
    def __init__(self, shape, activation):
        self.input = None
        self.output = None
        self.output_side_deltas = None
        self.input_side_deltas = None

    def forward_pass(self):
        raise NotImplementedError

    def backward_pass(self):
        raise NotImplementedError


class SimpleLinearLayer(Layer):
    def __init__(self, rows, cols):
        self.shape = (rows, cols)
        self.w = initialize_weight_array(rows, cols)

    def forward_pass(self):
        self.output = np.dot(self.input, self.w)
        return self.output

    def backward_pass(self, alpha, lam=0.0):
        self.input_side_deltas = np.dot(self.output_side_deltas, self.w.T)
        if lam:
            self.w += alpha * lam * self.w
        self.w += alpha * np.dot(self.input.T, self.output_side_deltas)
        return self.input_side_deltas


def initialize_weight_array(l, w, stddev=None, relu=False, sigma_cutoff=2.0):
    """
    Initializes a weight array with l rows and w columns.
    If stddev is not specified, default initialization is designed to create a variance of 1.0,
    meaning stddev is sqrt(1 / N_in). If the weight array is going to be used with relu
    activation, the default stddev will be sqrt(2 / N_in), since presumably half the neurons
    won't fire.
    sigma_cutoff determines the max number of stddevs away from 0 an initialized value can be.
    """
    if stddev is None:
        if relu:
            stddev = (2.0 / l) ** 0.5
        else:
            stddev = (1.0 / l) ** 0.5

    weights = []
    while len(weights) < l * w:
        new_rand_val = np.random.randn() * stddev
        if abs(new_rand_val) < sigma_cutoff * stddev:
            weights.append(new_rand_val)
    return np.array(weights).reshape(l, w)


def read_input(lines):
    all_input = [line.split() for line in lines.split('\n')]
    given_data_count = int(all_input[0][1])
    x = np.array([list(map(float, data[:-1])) for data in all_input[1: given_data_count + 1]])
    y = np.array([[float(data[-1])] for data in all_input[1: given_data_count + 1]])
    x_new = np.array([list(map(float, data[:-1])) for data in all_input[given_data_count + 2:]])
    y_new = np.array([[float(data[-1])] for data in all_input[given_data_count + 2:]])

    return x, y, x_new, y_new


input_text = "2 7\n\
0.18 0.89 109.85\n\
1.0 0.26 155.72\n\
0.92 0.11 137.66\n\
0.07 0.37 76.17\n\
0.85 0.16 139.75\n\
0.99 0.41 162.6\n\
0.87 0.47 151.77\n\
4\n\
0.49 0.18 105.22\n\
0.57 0.83 142.68\n\
0.56 0.64 132.94\n\
0.76 0.18 129.71"

x_train, y_train, x_test, y_test = read_input(input_text)
x_data = {'train': x_train, 'test': x_test}
y_data = {'train': y_train, 'test': y_test}

# print(x_test.shape)

lin_reg_network = Network([SimpleLinearLayer(2, 1)])
# print(lin_reg_network.feed_forward(x_test))

lin_reg_network.train(x_data, y_data, alpha=0.01, epochs=100000)

print(lin_reg_network.feed_forward(x_test))



