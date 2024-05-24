import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        max_values = np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(input_tensor - max_values)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, error_tensor):
        # E_n-1 = y_h * (E_n - sum of E*y_h for the batch)
        error = self.output * (error_tensor - np.sum(error_tensor * self.output, axis=1, keepdims=True))
        return error
