import numpy as np

class Activation_ReLU:
    """Rectified Linear activation. Only oututs values greater than 0
    """
    # Forward pass
    def forward(self, inputs):
        """Takes a list of input values and sets the class "output" variabe to a numpy array of 
        linear result <= 0

        Args:
            inputs (Numpy array): Numpy array containing numerical values.
        """
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    """Softmax activation. Exponentates and normalizes the values inputed
    """
    # Forward pass
    def forward(self, inputs):
        """Takes a batch of inputs and outputs the normalized exponential as probablilities

        Args:
            inputs (np.array): Batch of input values
        """
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities