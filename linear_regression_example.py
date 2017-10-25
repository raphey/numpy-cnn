__author__ = 'raphey'

import numpy as np
from generalized_nn import *


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
train = {'x': x_train, 'y_': y_train}
test = {'x': x_test, 'y_': y_test}

lin_reg_network = Network([FullyConnectedLayer(2, 1)])

train_regression_model(lin_reg_network, train, test, alpha=1.0, epochs=1000, verbose=True)
# Note that this example data, which came from hackerrank.com, has testing loss converge
# to zero presumably, because this example was made by selecting points from a random plane,
# after which noise was added only to the training points.
