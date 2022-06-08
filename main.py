import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense_layer import Layer_Dense
import matplotlib.pyplot as plt
import activation_functions as af
import loss_calulations as loss_calcs

nnfs.init()

# Create a dataset
X, y = spiral_data(samples=100, classes=3)

# Create a Dense layer with 2 input nodes and 3 output
dense1 = Layer_Dense(2,3)
# Create ReLU activation (to be used with dense layer 1)
activation1 = af.Activation_ReLU()

# Creating second dense layer with 3 input features as the data will be taken
# from the output of the previous layer here. Attribute it 3 output values
dense2 = Layer_Dense(3,3)

# Create a softmax activation to be used with this layer
activation2 = af.Activation_Softmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through an activation function.
# This takes values from the previous layers output.
activation1.forward(dense1.output)

# Make a forward pass through the second dense layer
# it takes the outputs of the previous activation function
dense2.forward(activation1.output)

# Make a forward pass through the softmax activation function taking the
# dense layers outpout
activation2.forward(dense2.output)

# Print the first few samples
print(activation2.output[:5])

# Perform a forward pass through activation function
# it takes the output of the second dense layer here and returns the loss

loss_function = loss_calcs.Loss_CategoricalCrossentrophy()
loss = loss_function.calculate(activation2.output, y)

# Print the loss value
print(f'loss: {loss}')