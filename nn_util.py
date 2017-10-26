__author__ = 'raphey'

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


def rough_print(num_arr):
    """
    Simple way to print a 784-length number array, outputting '.' for every cell == 0 and 'X' for cells > 0
    """
    new_shape = num_arr.reshape((28, 28))
    for row in new_shape:
        row_str = ""
        for entry in row:
            if entry > 0:
                row_str += 'X'
            else:
                row_str += '.'
        print(row_str)


def shuffle_data(data_obj, random_seed=0):
    """
    Given a data_obj with ['data'] and ['target] entries, shuffles them and returns them as separate arrays.
    """
    d = data_obj['data']
    t = data_obj['target'].reshape(-1, 1)

    return shuffle(d, t, random_state=random_seed)


def import_and_prepare_mnist_data(valid_portion=0.1, test_portion=0.1, flat=True):
    """
    Imports mnist data, shuffles it, and splits it into training, validation, and testing sets.

    If flat parameter is set to False, each image will be reshaped from (784) to (28 x 28 x 1), for convolution.

    training, validation, and testing are dicts with three keys each:
      'x': the image data
      'y_': the one-hot encoded labels
      'y_as_int': the labels as integers, for quick accuracy checking

    """

    mnist = fetch_mldata('MNIST original')
    data_size = len(mnist['data'])

    img_data, int_targets = shuffle_data(mnist)

    if not flat:
        img_data = img_data.reshape(-1, 1, 28, 28)

    scaled_data = img_data / 255.0

    int_targets = int_targets.astype(int)

    one_hots = one_hot_encode(int_targets)

    # Cutoff indices between training/validation and validation/testing
    validation_start = int((1.0 - valid_portion - test_portion) * data_size)
    testing_start = int((1.0 - test_portion) * data_size)

    train = {'x': scaled_data[:validation_start],
             'y_': one_hots[:validation_start],
             'y_as_int': int_targets[:validation_start]}

    valid = {'x': scaled_data[validation_start: testing_start],
             'y_': one_hots[validation_start: testing_start],
             'y_as_int': int_targets[validation_start: testing_start]}

    test = {'x': scaled_data[testing_start:],
            'y_': one_hots[testing_start:],
            'y_as_int': int_targets[testing_start:]}

    return train, valid, test


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


def one_hot_encode(targets):
    """
    One hot encodes targets. [4] --> [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    encoded_data = []
    for t in targets:
        new_t = np.zeros(10)
        new_t[int(t)] = 1.0
        encoded_data.append(new_t)
    return np.array(encoded_data)


def prediction_mse(y_actual, y_pred):
    """
    Returns mean-square error between actual y and predicted y.
    """
    return 0.5 * sum((y_actual[i] - y_pred[i]) ** 2 for i in range(0, len(y_actual)))


def prediction_cel(y_actual, y_pred):
    """
    Returns cross-entropy loss between actual y and predicted y.
    """
    if y_actual.ndim == 1:
        y_actual = [y_actual]
        y_pred = [y_pred]
    size = len(y_actual) * len(y_actual[0])
    return -1.0 / size * np.sum(y_actual * np.log(y_pred) + np.log(1.0 - y_pred) * (1.0 - y_actual))


def sigmoid(x):
    return np.ones(shape=x.shape) / (1.0 + np.exp(-x))


def soft_max(z):
    if z.ndim == 1:
        z = [z]
    exp_z = np.exp(z)
    sums = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / sums


def pad_image(img_array, top_pad, bottom_pad, left_pad, right_pad):
    """
    Pads the width and height dimensions of an image array or batch of image arrays
    with zeros, and returns a new padded array.
    img_array can be a single flat image with dimensions (height, width), an image
    with depth with dimensions (depth, height, width), or a batch of images with depth
    with dimensions (batch size, depth, height, width).
    """
    img_height = img_array.shape[-2]
    img_width = img_array.shape[-1]

    # Sets the correct shape for the padded version for 2, 3, or 4 dimensions
    padded_shape = list(img_array.shape)
    padded_shape[-2] += top_pad + bottom_pad
    padded_shape[-1] += left_pad + right_pad
    padded_img = np.zeros(padded_shape)

    if len(img_array.shape) == 2:
        padded_img[top_pad: top_pad + img_height, left_pad: left_pad + img_width] = img_array
    elif len(img_array.shape) == 3:
        padded_img[:, top_pad: top_pad + img_height, left_pad: left_pad + img_width] = img_array
    else:
        padded_img[:, :, top_pad: top_pad + img_height, left_pad: left_pad + img_width] = img_array

    return padded_img


def flat_img_to_conv_stack(img, window_size, stride):
    """
    Given a flat image, returns a convolutional stack obtained by passing a square
    window across the image (left to right along the top, then next row down, etc).
    Each window is unrolled into a single 1-D row, and the stack has dimensions
    number_of_windows x window_size^2.
    """
    img_height, img_width = img.shape
    unrolled_window_size = window_size ** 2
    conv_stack = []

    for i in range(0, img_height - window_size + 1, stride):
        for j in range(0, img_width - window_size + 1, stride):
            conv_stack.append(img[i: i + window_size, j:j + window_size].reshape(unrolled_window_size))

    return np.array(conv_stack)