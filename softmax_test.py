layer_outputs = [4.8, 1.21, 2.385]
from cmath import exp
import math

from numpy import append

# Exponentiate the numbers Y = e^x
E = math.e

# For each value in a vector, calculate the exponentiated value
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print('Exponentiated values:')
print(exp_values)

# Normalize the values
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values: {}'.format(sum(norm_values)))