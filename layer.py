inputs = [1, 2, 3, 2.5]
weights =  [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# Output for the current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0

    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this inout by its associated weight
        # and add the neuron's output variable
        neuron_output += n_input*weight
    
    # Add the bias
    neuron_output += neuron_bias
    # Out neuron's result to the layer's output list
    layer_outputs.append(neuron_output)


print(layer_outputs)