import math
import numpy as np

# An example output from the out layer of the neural network
softmax_example_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_example_output[0]) * target_output[0] +
        math.log(softmax_example_output[1]) * target_output[1] +
        math.log(softmax_example_output[2]) * target_output[2])

print(loss)

num = 5.2
log_num = np.log(num)

print('''Testing log({}) = {}. \n
    e ** num = {}'''.format(num, log_num, math.e**log_num))