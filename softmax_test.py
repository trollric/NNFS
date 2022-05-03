layer_outputs = [4.8, 1.21, 2.385]
import numpy as np

# With numpy
# Exoinentiate the numbers
exp_values = np.exp(layer_outputs)
print(exp_values)

# Normalize the values
norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values: {}'.format(sum(norm_values)))