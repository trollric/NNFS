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

softmax_output = [  [0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]]

class_targets =[0, 1, 1]

for target_index, distribution in zip(class_targets, softmax_output):
    print(distribution[target_index])

print('Test with numpy array')

softmax_output = np.array([  [0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]])

class_targets =[0, 1, 1]

print(softmax_output[[0, 1, 2], class_targets])

print(softmax_output[range(len(softmax_output)), class_targets])