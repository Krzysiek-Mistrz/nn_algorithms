import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        #u nas macierz 4x3 czyli 3 neurony i 4 dane w 1 seri wejsciowych
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        min_values = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(min_values)
        propabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = propabilities


X, y = spiral_data(samples=100, classes=3)

layer1 = LayerDense(2, 3)
activation1 = ActivationReLU()
layer2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])