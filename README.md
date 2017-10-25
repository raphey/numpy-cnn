## Making a convolutional neural network with NumPy


The purpose of this project is to build a CNN using only NumPy.

The current state of the project is a generalized non-convolutional network that can classify MNIST with ~98% accuracy. It includes leaky ReLU activation, L2 regularization, and dropout.

This was influenced by [Michael Nielsen's amazing eBook](http://neuralnetworksanddeeplearning.com/).

### To-do

- Add convolutional layer with stride, same padding, no pooling
- Train an MNIST classifier to above 99%
- Implement saving/loading of model?
- Add visualization of hidden layers?
- Try on CIFAR-10?
- Batch normalization? (might run into trouble with the linear network structure)

### File descriptions
- [four_layer_nn.py](four_layer_nn.py): MNIST classifier network with two hidden layers, 98.1% accuracy after 200 epochs of training (~2 hours)

_________________________________________