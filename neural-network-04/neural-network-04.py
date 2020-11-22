class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, neurons)
        self.baises = np.zeros((1, n_neurons))

    #forward pass
    def forward(self, inputs):
        pass
