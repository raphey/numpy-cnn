__author__ = 'raphey'

import numpy as np
from sklearn.datasets import fetch_mldata


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
    np.random.seed(seed=random_seed)
    d = data_obj['data']
    t = data_obj['target'].reshape(-1, 1)
    joint_arr = np.hstack((d, t))
    np.random.shuffle(joint_arr)
    joint_arr = joint_arr.T
    new_d = joint_arr[:784].T
    new_t = joint_arr[-1].T
    return new_d, new_t


def import_and_prepare_data(valid_portion=0.1, test_portion=0.1, flat=True):
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
        img_data = img_data.reshape(-1, 28, 28, 1)

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


def make_prediction(x):
    h1_out = np.dot(x, w1) + b1
    sig_h1 = sigmoid(h1_out)
    h2_out = np.dot(sig_h1, w2) + b2
    sig_h2 = sigmoid(h2_out)
    y_hat = np.dot(sig_h2, w3)
    return y_hat.argmax()


def accuracy(imgs, int_labels):
    correct = 0.0
    for img, int_label in zip(imgs, int_labels):
        y_pred = make_prediction(img)
        if y_pred == int_label:
            correct += 1
    return correct / len(imgs)


def sigmoid(x):
    return np.ones(shape=x.shape) / (1.0 + np.exp(-x))


def soft_max(z):
    if z.ndim == 1:
        z = [z]
    exp_z = np.exp(z)
    sums = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / sums


def test_and_show_random_digit():
    j = np.random.randint(len(testing['x']))
    x = testing['x'][j]
    y = testing['y_as_int'][j]
    h1_out = np.dot(x, w1) + b1
    sig_h1 = sigmoid(h1_out)
    h2_out = np.dot(sig_h1, w2) + b2
    sig_h2 = sigmoid(h2_out)
    z_l = np.dot(sig_h2, w3)
    a_l = soft_max(z_l)

    print("---------------------------------")
    print("Hand-written digit:")
    rough_print(x)
    print("Softmax predictions:")
    predictions = list(zip(range(10), a_l[0]))
    predictions.sort(reverse=True, key=lambda a: a[1])
    for k in range(0, 3):
        print("  {}: \t {:>5.3f}".format(predictions[k][0], predictions[k][1]))
    print("Actual value:", y)
    print()


def train_model(alpha=0.01, epochs=100, batch_size=10, lam=0.1):
    global w1, w2, w3, b1, b2
    num_batches = training_size // batch_size
    for i in range(epochs):
        # (Switched to using training accuracy
        # training_loss = 0.0
        correct_count = 0
        for j in range(num_batches):
            start_index = j * batch_size
            end_index = start_index + batch_size
            x = training['x'][start_index: end_index]
            y = training['y_'][start_index: end_index]

            h1_out = np.dot(x, w1) + b1
            sig_h1 = sigmoid(h1_out)
            h2_out = np.dot(sig_h1, w2) + b2
            sig_h2 = sigmoid(h2_out)
            z_l = np.dot(sig_h2, w3)
            a_l = soft_max(z_l)

            for ii in range(len(x)):
                if list(y[ii]).index(1.0) == list(a_l[ii]).index(max(a_l[ii])):
                    correct_count += 1

            y_diff = y - a_l

            delta_h2o = np.dot(y_diff, w3.T) * sig_h2 * (np.ones(shape=sig_h2.shape) - sig_h2)
            delta_h1o = np.dot(delta_h2o, w2.T) * sig_h1 * (np.ones(shape=sig_h1.shape) - sig_h1)

            w1 += -alpha * lam / training_size * w1
            w1 += alpha / batch_size * np.dot(x.T, delta_h1o)

            w2 += -alpha * lam / training_size * w2
            w2 += alpha / batch_size * np.dot(sig_h1.T, delta_h2o)

            w3 += -alpha * lam / training_size * w3
            w3 += alpha / batch_size * np.dot(sig_h2.T, y_diff)

            b1 += alpha / batch_size * delta_h1o.sum(axis=0)
            b2 += alpha / batch_size * delta_h2o.sum(axis=0)

        print("Epoch {:>3}\t Avg epoch training acc: {:>5.3f}\t Validation acc: {:>5.3f} ".format
              (i + 1, correct_count / training_size, accuracy(validation['x'], validation['y_as_int'])))

    print("Final training accuracy: {:>5.3f}".format(accuracy(training['x'], training['y_as_int'])))
    print("Final test accuracy: {:>5.3f}".format(accuracy(testing['x'], testing['y_as_int'])))


if __name__ == "__main__":

    training, validation, testing = import_and_prepare_data(0.1, 0.1)
    training_size = len(training['x'])

    # define layer sizes
    l1 = 784
    l2 = 250
    l3 = 50
    l4 = 10

    # initialize weights
    w1 = initialize_weight_array(l1, l2)
    w2 = initialize_weight_array(l2, l3)
    w3 = initialize_weight_array(l3, l4)

    # initialize biases
    b1 = np.zeros(l2)
    b2 = np.zeros(l3)

    train_model(batch_size=64, epochs=10)

    for _ in range(10):
        test_and_show_random_digit()
