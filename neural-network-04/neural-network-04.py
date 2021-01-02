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

class Activation_ReLU():

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalize probabilites
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims = True)
        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense Layer with 2 inputs features and 3 output values
dense1= Layer_Dense(2, 3)
# Create RELU activation ( to be used with Dense Layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 inputs features ( from prevous layer)
# and 3 ouputs values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of the first dense layer
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)


print(activation2.output[:5])
