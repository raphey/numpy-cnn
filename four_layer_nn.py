__author__ = 'raphey'

import numpy as np
from nn_util import initialize_weight_array, sigmoid, soft_max, rough_print, import_and_prepare_mnist_data


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

    training, validation, testing = import_and_prepare_mnist_data(0.1, 0.1)
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
