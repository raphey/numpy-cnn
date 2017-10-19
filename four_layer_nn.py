__author__ = 'raphey'

import numpy as np
from sklearn.datasets import fetch_mldata


"""L2 regularization with a 4-layer network. With 200 epochs, batch size 10, test accuracy gets to 98.1% with no L2reg.
With lambda = 0.1, still 98.1%, but slightly worse. Doesn't seem like a situation in which regularization is needed.
"""


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


def shuffle_data(data_obj):
    """
    Given a data_obj with ['data'] and ['target] entries, shuffles them and returns them as separate arrays.
    """
    d = data_obj['data']
    t = data_obj['target'].reshape(-1, 1)
    joint_arr = np.hstack((d, t))
    np.random.shuffle(joint_arr)
    joint_arr = joint_arr.T
    new_d = joint_arr[:784].T
    new_t = joint_arr[-1].T
    return new_d, new_t


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
    return -1.0 / size * np.sum(y_actual * np.log(y_pred) + (1.0 - y_actual) * np.log(1.0 - y_pred))


def make_prediction(x):
    h1_out = np.dot(x, W1) + b1
    sig_h1 = sigmoid(h1_out)
    h2_out = np.dot(sig_h1, W2) + b2
    sig_h2 = sigmoid(h2_out)
    y_hat = np.dot(sig_h2, W3)
    return y_hat.argmax()


def accuracy(imgs, int_labels):
    correct = 0.0
    for img, int_label in zip(imgs, int_labels):
        y_pred = make_prediction(img)
        if y_pred == int_label:
            correct += 1
    return correct / len(imgs)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_max(z):
    if z.ndim == 1:
        z = [z]
    exp_z = np.exp(z)
    sums = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / sums


def test_and_show_random_digit():
    i = np.random.randint(len(test_data))
    x = test_data[i]
    y = test_targets_as_int[i]
    h1_out = np.dot(x, W1) + b1
    sig_h1 = sigmoid(h1_out)
    z_L = np.dot(sig_h1, W2)
    a_L = soft_max(z_L)
    print("---------------------------------")
    print("Hand-written digit:")
    rough_print(x)
    print("Softmax predictions:")
    predictions = list(zip(range(10), a_L[0]))
    predictions.sort(reverse=True, key=lambda x: x[1])
    for i in range(0, 3):
        print("  %s: \t %.3f" % predictions[i])
    print("Actual value:", y)
    print()


"""
mnist is an sklearn object with attributes 'data', 'target' and two others, 'COL_NAMES' and 'DESCR'
mnist['target'] is a list of 70000 correct number labels, as floats.
mnist['data'] is the image data for the 70000 numbers, each of which is a length 784 vector of integers from 0 to 255.
The images are 28 by 28, with the first 28 integers in the length 784 vector constituting the first row.
"""
mnist = fetch_mldata('MNIST original')


mnist_data, mnist_targets = shuffle_data(mnist)

mnist_targets = mnist_targets.astype(int)
one_hot_targets = one_hot_encode(mnist_targets)
scaled_mnist_data = mnist_data / 255.0


test_portion = 0.1          # Portion of data reserved for testing
validation_portion = 0.1    # Portion of non-testing data reserved for validation

test_cutoff = int(len(scaled_mnist_data) * (1 - test_portion))      # index of first piece of data/target for testing
training_n = int(test_cutoff * (1 - validation_portion))   # index of first piece of data/target for validation


# Splitting data into training, validation, and testing
training_targets = one_hot_targets[:training_n]
training_targets_as_int = mnist_targets[:training_n]
training_data = scaled_mnist_data[:training_n]

validation_targets = one_hot_targets[training_n:test_cutoff]
validation_targets_as_int = mnist_targets[training_n:test_cutoff]
validation_data = scaled_mnist_data[training_n:test_cutoff]

test_targets = one_hot_targets[test_cutoff:]
test_targets_as_int = mnist_targets[test_cutoff:]
test_data = scaled_mnist_data[test_cutoff:]


# define layer sizes
l1 = 784
l2 = 250
l3 = 50
l4 = 10

# initialize weights
W1 = initialize_weight_array(l1, l2)
W2 = initialize_weight_array(l2, l3)
W3 = initialize_weight_array(l3, l4)

# initialize biases
b1 = np.zeros(l2)
b2 = np.zeros(l3)

alpha = 0.01

lam = 0.1

N = 10
batch_size = 10
num_batches = training_n // batch_size

for i in range(N):
    training_loss = 0.0
    correct_count = 0
    for j in range(num_batches):
        start_index = j * batch_size
        end_index = start_index + batch_size
        x = training_data[start_index: end_index]
        y = training_targets[start_index: end_index]

        h1_out = np.dot(x, W1) + b1
        sig_h1 = sigmoid(h1_out)
        h2_out = np.dot(sig_h1, W2) + b2
        sig_h2 = sigmoid(h2_out)
        z_L = np.dot(sig_h2, W3)
        a_L = sigmoid(z_L)

        for ii in range(len(x)):
            if list(y[ii]).index(1.0) == list(a_L[ii]).index(max(a_L[ii])):
                correct_count += 1

        y_diff = y - a_L

        delta_h2o = np.dot(y_diff, W3.T) * sig_h2 * (1 - sig_h2)
        delta_h1o = np.dot(delta_h2o, W2.T) * sig_h1 * (1 - sig_h1)

        W1 += -alpha * lam / training_n * W1
        W1 += alpha / batch_size * np.dot(x.T, delta_h1o)

        W2 += -alpha * lam / training_n * W2
        W2 += alpha / batch_size * np.dot(sig_h1.T, delta_h2o)

        W3 += -alpha * lam / training_n * W3
        W3 += alpha / batch_size * np.dot(sig_h2.T, y_diff)

        b1 += alpha / batch_size * delta_h1o.sum(axis=0)
        b2 += alpha / batch_size * delta_h2o.sum(axis=0)

    print("Epoch {:>3}\t Validation acc: {:>5.3f} \t Training acc: {:>5.3f}".format
          (i, accuracy(validation_data, validation_targets_as_int), correct_count / training_n, 4))


print("Final training accuracy: {:>5.3f}".format(accuracy(training_data, training_targets_as_int)))
print("Final test accuracy: {:>5.3f}".format(accuracy(test_data, test_targets_as_int)))
