## Making a convolutional neural network with NumPy


The purpose of this project is to build a CNN using only NumPy, plus some other convenient tools for manipulating data. The current state of the project is a MNIST non-convolutional network with two hidden layers and a testing accuracy of 98%.

This will undoubtedly end up being influenced by [Michael Nielsen's amazing eBook](http://neuralnetworksanddeeplearning.com/), but since my goal is to learn as much as possible, I'm going to try to make something distinct.

### To-do

- Add documentation to four_layer_nn
- Improve data loading, including random seed for consistent shuffling
- Put general utility functions in separate module
- Make a generalized non-convolutional network (weights and biases as lists, presumably)
- Implement saving/loading of model


### File descriptions
- [four_layer_nn.py](four_layer_nn.py): MNIST classifier network with two hidden layers, 98.1% accuracy after 200 epochs of training (~2 hours)

_________________________________________