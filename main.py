import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense_layer import Layer_Dense
import matplotlib.pyplot as plt
from activation_functions import Activation_ReLU as ReLU

nnfs.init()

# Create a dataset
X, y = spiral_data(samples=100, classes=3)

# Create a Dense layer with 2 input nodes and 3 output
dense1 = Layer_Dense(2,3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through an activation function.
# This takes values from the previous layers output.

activation1 = ReLU()
activation1.forward(dense1.output)

# Print the first 5 outputs
print(activation1.output[:5])