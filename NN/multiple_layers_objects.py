import numpy as np
np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        #u nas macierz 4x3 czyli 3 neurony i 4 dane w 1 seri wejsciowych
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = LayerDense(4, 3)
layer2 = LayerDense(3, 5)

layer1_output = layer1.forward(X)
layer2_output = layer2.forward(layer1_output)