## Making a convolutional neural network with NumPy


The purpose of this project is to build a modular CNN using only NumPy. So far, the project includes the following features:

- Generalized network class created with layer sizes as parameters
- Sigmoid and leaky ReLU activation
- Dropout and L2 regularization
- Convolutional layer without any regularization or padding

The best accuracy I've gotten so far is 98.7%, using two convolutional layers and a fully connected layer with dropout. The goal is to get something over 99%.

This was influenced by [Michael Nielsen's amazing eBook](http://neuralnetworksanddeeplearning.com/), and I also found [this explanation of CNN backprop](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/) really helpful.

### To-do

- Add lambda to convolution layer?
- Add same padding option to convolution layer
- Use flexible parameters to make CNN, vs hardcoded layers
- Train an MNIST classifier to above 99%
- Implement saving/loading of model?
- Add visualization of hidden layers?
- Try on CIFAR-10?
- Batch normalization? (might run into trouble with the linear network structure)

### File descriptions
- [generalized_nn.py](generalized_nn.py): MNIST classifier network using flexible network and layer classes. Gets up to 98.4% after 200 epochs.
- [nn_util.py](nn_util.py): General utility functions, including functions for importing and preparing MNIST data
- [linear_regression_example.py](linear_regression_example.py): Example of general network being used for a simple linear regression
- [four_layer_nn.py](four_layer_nn.py): Old MNIST classifier network with two hand-made, non-modular hidden layers, 98.1% accuracy after 200 epochs.

_________________________________________