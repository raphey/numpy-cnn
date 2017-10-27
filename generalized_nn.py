__author__ = 'raphey'

import numpy as np
from nn_util import import_and_prepare_mnist_data
import warnings


class Network(object):
    """
    Base class for a neural net.

    self.layers is a list of layers going in order from input to output.
    With this structure, activation functions count as separate layers.

    self.feed_forward uses a series of layer forward_pass methods to go
    from an input into an output and also sets the input and output
    properties of the corresponding layers.

    self.feed_backward uses a series of layer backward_pass methods to
    go from output deltas backwards through the network, and modifies
    layers if applicable

    self.train trains the network.
    """
    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, x_in):
        self.layers[0].input = x_in
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].input = self.layers[i].forward_pass()
        self.layers[-1].forward_pass()
        return self.layers[-1].output

    def feed_backward(self, delta_y_out, backprop_params):
        self.layers[-1].output_side_deltas = delta_y_out
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i - 1].output_side_deltas = self.layers[i].backward_pass(backprop_params)
        self.layers[0].backward_pass(backprop_params)

    @staticmethod
    def mse_cost(y_predicted, y_actual):
        """
        Returns total mean-square error for predicted values and actual values.
        """
        return ((y_actual - y_predicted)**2).mean()


class Classifier(Network):
    """
    Classifier network, with an accuracy method and a static cross-entropy loss method
    """

    def accuracy(self, x_input, labels_as_values):
        correct = 0.0
        all_output = self.feed_forward(x_input)
        for logit, label in zip(all_output, labels_as_values):
            if np.argmax(logit) == label:
                correct += 1
        return correct / len(x_input)

    @staticmethod
    def cross_entropy_cost(y_predicted, y_actual):
        """
        Returns total mean-square error for predicted values and actual values.
        Pads predicted values very close to 0.0 or 1.0 to avoid overflowing cost
        """
        epsilon = 1e-12
        y_predicted[y_predicted < epsilon] = epsilon
        y_predicted[y_predicted > 1 - epsilon] = 1 - epsilon

        size = len(y_actual) * len(y_actual[0])

        ce_cost = -1.0 / size * np.sum(y_actual * np.log(y_predicted) + np.log(1.0 - y_predicted) * (1.0 - y_actual))

        return ce_cost


class Layer(object):
    """
    Base class for layers, which will include matrices, activation functions, and
    convolution layers.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.output_side_deltas = None
        self.input_side_deltas = None

    def forward_pass(self):
        raise NotImplementedError

    def backward_pass(self, backprop_params):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    """
    Fully connected layer in which input is multiplied by a trainable weight matrix
    """

    def __init__(self, rows, cols, relu=False):
        super().__init__()
        self.shape = (rows, cols)
        self.w = initialize_weight_array(rows, cols, relu=relu)
        self.b = np.zeros(shape=(1, cols))

    def forward_pass(self):
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    def backward_pass(self, backprop_params):
        alpha_adj, lam = backprop_params
        self.input_side_deltas = np.dot(self.output_side_deltas, self.w.T)
        if lam:
            self.w *= (1.0 - lam * alpha_adj)
        self.w += alpha_adj * np.dot(self.input.T, self.output_side_deltas)
        self.b += alpha_adj * self.output_side_deltas.sum(axis=0)
        return self.input_side_deltas


class SigmoidLayer(Layer):
    """
    Sigmoid activation layer. Input and output have the same shape, as do the input-side and
    output-side deltas.
    """
    def forward_pass(self):
        self.output = 1.0 / (1.0 + np.exp(-self.input))
        return self.output

    def backward_pass(self, backprop_params):
        # Backprop parameters are not used.
        self.input_side_deltas = self.output_side_deltas * self.output * (1.0 - self.output)
        return self.input_side_deltas


class SoftmaxLayer(Layer):
    """
    Softmax activation layer, to be used right before output. Backprop is skipped entirely,
    under the assumption that this will be used with cross-entropy loss.
    """
    def forward_pass(self):
        exp_z = np.exp(self.input)
        sums = np.sum(exp_z, axis=1, keepdims=True)
        self.output = exp_z / sums
        return self.output

    def backward_pass(self, backprop_params):
        # Backprop parameters are not used
        self.input_side_deltas = self.output_side_deltas
        return self.input_side_deltas


class LReLULayer(Layer):
    """
    Leaky ReLU activation layer. Input and output have the same shape, as do the input-side and
    output-side deltas.
    """
    def __init__(self, a=0.01):
        super().__init__()
        self.a = a

    def forward_pass(self):
        self.output = np.maximum(self.input, self.a * self.input)
        return self.output

    def backward_pass(self, backprop_params):
        # Backprop parameters are not used.
        pos_boolean = self.input >= 0
        self.input_side_deltas = self.a * self.output_side_deltas[:]
        self.input_side_deltas[pos_boolean] = self.output_side_deltas[pos_boolean]

        return self.input_side_deltas


class FullyConnectedLayerWithDropout(Layer):
    """
    Fully connected layer in which input is multiplied by a trainable weight matrix, with
    dropout that can be turned on or off.
    """

    def __init__(self, rows, cols, keep_prob, relu=False):
        super().__init__()
        self.shape = (rows, cols)
        self.w = initialize_weight_array(rows, cols, relu=relu)
        self.b = np.zeros(shape=(1, cols))
        self.keep_prob = keep_prob
        self.dropout_on = False
        self.keep_mask = None

    def forward_pass(self):
        adjusted_weight = self.w.copy()

        if self.dropout_on:
            self.keep_mask = np.random.binomial([np.ones(self.w.shape)], self.keep_prob)[0] * (1.0 / self.keep_prob)
            adjusted_weight *= self.keep_mask

        self.output = np.dot(self.input, adjusted_weight) + self.b
        return self.output

    def backward_pass(self, backprop_params):
        if not self.dropout_on:
            warnings.warn("Warning: Backprop is being run without dropout, which is probably an error.")
        alpha_adj, _ = backprop_params   # Not using L2 regularization

        adjusted_weight = self.w.copy()

        if self.dropout_on:
            adjusted_weight *= self.keep_mask

        self.input_side_deltas = np.dot(self.output_side_deltas, adjusted_weight.T)

        weight_delta = alpha_adj * np.dot(self.input.T, self.output_side_deltas)

        if self.dropout_on:
            weight_delta *= self.keep_mask

        self.w += weight_delta
        self.b += alpha_adj * self.output_side_deltas.sum(axis=0)

        return self.input_side_deltas


class ConvolutionLayer(Layer):
    """
    Fully connected layer in which input is multiplied by a trainable weight matrix
    """

    def __init__(self, channels_out, channels_in, window_size, stride, pad=False, relu=True):
        super().__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.window_size = self.window_size
        self.stride = stride
        self.pad = pad

        self.shape_4d = (channels_out, channels_in, window_size, window_size)
        self.filters_2d = initialize_weight_array(channels_in * window_size**2, channels_out, relu=relu)
        self.filters_4d = self.filters_2d.T.reshape(self.shape_4d)
        self.b = np.zeros(shape=(1, channels_out))

        self.batch_size = None
        self.reshaped_input = None
        self.output_height = None
        self.output_width = None

    def forward_pass(self):
        self.reshaped_input = img_batch_to_conv_stacks(self.input, self.window_size, self.stride)
        self.batch_size = self.input.shape[0]

        reshaped_output = (np.dot(self.reshaped_input, self.filters_2d) + self.b)
        self.output = reshaped_output.T.reshape(self.channels_out, self.batch_size,
                                                self.output_height, self.output_width).transpose(1, 0, 2, 3)
        return self.output

    def backward_pass(self, backprop_params):
        alpha_adj, _ = backprop_params   # Ignoring lambda

        reshaped_output_side_deltas = self.output_side_deltas.transpose(1, 0, 2, 3).reshape(self.channels_out, -1).T

        self.input_side_deltas = np.dot(reshaped_output_side_deltas, self.filters_2d.T)

        self.filters_2d += alpha_adj * np.dot(self.reshaped_input.T, reshaped_output_side_deltas)
        self.filters_4d = self.filters_2d.T.reshape(self.shape_4d)

        self.b += alpha_adj * self.output_side_deltas.sum(axis=(0, 2, 3))

        return self.input_side_deltas

    def img_batch_to_conv_stacks(self):
        """
        Takes the current input, a batch of images with depth, and sets the reshape_input property to be series
        of convolutional stacks obtained by passing a square prism window with matching depth across each image
        (left to right along the top, then next row down, etc, then same for remaining channels, then next image).
        Each window prism is unrolled into a single 1-D row, and the stack array has dimensions
        (batch size * number_of_windows) by (window_size^2 * depth).
        """
        batch_size, img_depth, img_height, img_width = self.input.shape
        unrolled_window_size = self.window_size ** 2 * self.img_depth

        self.output_height = (img_height - self.window_size) // self.stride + 1
        self.output_width = (img_width - self.window_size) // self.stride + 1

        conv_stack = []

        for k in range(0, batch_size):
            for i in range(0, img_height - self.window_size + 1, self.stride):
                for j in range(0, img_width - self.window_size + 1, self.stride):
                    conv_stack.append(self.input[k, :, i: i + self.window_size, j:j + self.window_size].reshape(
                                      unrolled_window_size))

        assert(conv_stack.shape[0] == batch_size * self.output_height * self.output_width)

        return np.array(conv_stack)


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


def train_regression_model(regression_net, train, test, alpha, epochs, lam=0.0, verbose=False):
    """
    Training tool for regressions--simpler than classification tool, currently not using
    validation or batches.
    """
    print("Training network with alpha={}, lambda={} for {} epochs...".format(alpha, lam, epochs))
    x_training, x_testing = train['x'], test['x']
    y_training, y_testing = train['y_'], test['y_']

    training_size = x_training.shape[0]

    for e in range(1, epochs + 1):
        delta_y = y_training - regression_net.feed_forward(x_training)
        regression_net.feed_backward(delta_y, [alpha / training_size, lam])
        if verbose and e % 100 == 0:
            print("Epoch {:>3}\t Training loss: {:>5.3f}".format
                  (e, regression_net.mse_cost(y_predicted=regression_net.layers[-1].output, y_actual=y_training)))
    print("Training complete. Testing loss: {:>5.3f}".format
          (regression_net.mse_cost(y_predicted=regression_net.feed_forward(x_testing), y_actual=y_testing)))


def train_classifier_model(classifier, train, valid, test, alpha, batch_size, epochs,
                           lam=0.0, dropout_model=False, verbose=False):

    print("Training network with alpha={}, lambda={}, batch size={} for {} epochs...".format(
          alpha, lam, batch_size, epochs))

    x_training, x_validation, x_testing = train['x'], valid['x'], test['x']
    y_training_int, y_validation_int, y_testing_int = train['y_as_int'], valid['y_as_int'], test['y_as_int']
    y_training_one_hot, y_validation_one_hot, y_testing_one_hot = train['y_'], valid['y_'], test['y_']

    num_batches = len(x_training) // batch_size

    for e in range(1, epochs + 1):
        training_loss = 0.0
        for j in range(num_batches):
            start_index = j * batch_size
            end_index = start_index + batch_size
            x = x_training[start_index: end_index]
            y_ = y_training_one_hot[start_index: end_index]

            if dropout_model:
                set_dropout_boolean(classifier, True)

            delta_y = y_ - classifier.feed_forward(x)

            classifier.feed_backward(delta_y, (alpha / num_batches, lam))

            if dropout_model:
                set_dropout_boolean(classifier, False)

            training_loss += classifier.cross_entropy_cost(y_predicted=classifier.layers[-1].output, y_actual=y_)

        if verbose and e % 5 == 0:
            print("Epoch {:>3}\t Training loss: {:>5.3f}\t Validation acc: {:>5.3f}".format
                  (e, training_loss / num_batches, classifier.accuracy(x_validation, y_validation_int)))

    print("Training complete. Testing loss: {:>5.3f} \t Testing accuracy: {:>5.3f}".format
          (classifier.cross_entropy_cost(y_predicted=classifier.feed_forward(x_testing), y_actual=y_testing_one_hot),
           classifier.accuracy(x_testing, y_testing_int)))


def make_classifier(layer_sizes):
    """
    Returns a classifier object with the specified fully connected layer sizes.
    Each fully connected layer except for the last is followed by a sigmoid
    activation layer. Last fully connected layer is followed by a softmax layer.
    For an MNIST network layer sizes might be something like [784, 150, 25, 10].
    """
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(SigmoidLayer())
    layers.append(FullyConnectedLayer(layer_sizes[-2], layer_sizes[-1]))
    layers.append(SoftmaxLayer())
    return Classifier(layers)


def make_lrelu_classifier(layer_sizes):
    """
    Returns a classifier object with the specified fully connected layer sizes.
    Each fully connected layer except for the last is followed by a LReLU
    activation layer. Last fully connected layer is followed by a softmax layer.
    For an MNIST network layer sizes might be something like [784, 150, 25, 10].
    """
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1], relu=True))
        layers.append(LReLULayer())
    layers.append(FullyConnectedLayer(layer_sizes[-2], layer_sizes[-1]))
    layers.append(SoftmaxLayer())
    return Classifier(layers)


def make_lrelu_classifier_with_dropout(layer_sizes, keep_prob):
    """
    Returns a classifier object with the specified fully connected layer sizes.
    Each fully connected layer except for the last is followed by a LReLU
    activation layer. Last fully connected layer is followed by a softmax layer.
    For an MNIST network layer sizes might be something like [784, 150, 25, 10].
    """
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(FullyConnectedLayerWithDropout(layer_sizes[i], layer_sizes[i + 1], keep_prob, relu=True))
        layers.append(LReLULayer())
    layers.append(FullyConnectedLayerWithDropout(layer_sizes[-2], layer_sizes[-1], keep_prob))
    layers.append(SoftmaxLayer())
    return Classifier(layers)


def set_dropout_boolean(network, dropout_boolean):
    for layer in network.layers:
        if type(layer).__name__ == 'FullyConnectedLayerWithDropout':
            layer.dropout_on = dropout_boolean


if __name__ == "__main__":
    training, validation, testing = import_and_prepare_mnist_data(0.1, 0.1)

    classifier_network = make_lrelu_classifier_with_dropout(layer_sizes=[784, 250, 50, 10], keep_prob=0.80)

    train_classifier_model(classifier_network, training, validation, testing, alpha=0.1, batch_size=64,
                           epochs=200, lam=0.00, dropout_model=True, verbose=True)