layer_outputs = [4.8, 1.21, 2.385]
import math

from numpy import append

# Exponentiate the numbers Y = e^x
E = math.e

# For each value in a vector, calculate the exponentiated value
exp_values = []

for output in layer_outputs:
    exp_values,append(E**output)

print('Exponentiated values:')
print(exp_values)