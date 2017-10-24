## Making a convolutional neural network with NumPy


The purpose of this project is to build a CNN using only NumPy, plus some other convenient tools for manipulating data.

The current state of the project is a generalized non-convolutional network that can classify MNIST with ~98% accuracy.

This will undoubtedly end up being influenced by [Michael Nielsen's amazing eBook](http://neuralnetworksanddeeplearning.com/), but since my goal is to learn as much as possible, I'm going to try to make something distinct.

### To-do

- Implement saving/loading of model
- Implement dropout
- Convolutional network
- Add visualization of hidden layers?
- Batch normalization? (Does this not work with the linear network structure?)

### File descriptions
- [four_layer_nn.py](four_layer_nn.py): MNIST classifier network with two hidden layers, 98.1% accuracy after 200 epochs of training (~2 hours)

_________________________________________