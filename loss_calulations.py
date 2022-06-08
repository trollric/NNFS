import numpy as np

# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Caclulate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss values
        return data_loss

# Cross entropy loss
class Loss_CategoricalCrossentrophy(Loss):

    # Forward pass
    def forward(self, y_prediction, y_true):

        # Number of samples in the batch
        samples = len(y_prediction)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)

        # Probabilities for target values
        # Only if categorical labels
        if(len(y_true.shape) == 1):
            correct_confidences = y_prediction_clipped[
                range(samples), y_true
            ]
        
        # Mask values - only for one-hot encoded labels
        elif(len(y_true.shape) == 2):
            correct_confidences = np.sum(
                y_prediction_clipped*y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods