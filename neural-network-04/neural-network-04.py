import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))

    #forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises

X, y = spiral_data(samples=100, classes=3)
dense1= Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])
