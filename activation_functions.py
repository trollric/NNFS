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