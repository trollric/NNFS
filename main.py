import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense_layer import Layer_Dense
import matplotlib.pyplot as plt

nnfs.init()

# Create a dataset
X, y = spiral_data(samples=100, classes=3)

# Create a Dense layer with 2 input nodes and 3 output
dense1 = Layer_Dense(2,3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Print the first 5 outputs
print(dense1.outputs[:5])