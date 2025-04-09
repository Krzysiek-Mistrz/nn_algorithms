import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

X, y = spiral_data(100, 3)

#https://cs231n.github.io/neural-networks-case-study/
"""def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y"""

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

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)