import numpy as np

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# Let's optimize and test backpropagation here
# ReLu activation - simulates derivative with respect to inputs
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0 ] = 0


# Dense Layers
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)

# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)

# dbiases - sum values, do this over samples
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters

weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)


