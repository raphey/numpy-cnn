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

    def feed_backward(self, delta_y_out):
        self.layers[-1].output_side_deltas = delta_y_out
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i - 1].output_side_deltas = self.layers[i].backward_pass()

    def train(self, x_data, y_data, alpha, epochs, verbose=False):
        x_training, x_testing = x_data['train'], x_data['test']
        y_training, y_testing = y_data['train'], y_data['test']
        for e in range(1, epochs + 1):
            delta_y = y_training - self.feed_forward(x_training)
            self.feed_backward(delta_y)


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