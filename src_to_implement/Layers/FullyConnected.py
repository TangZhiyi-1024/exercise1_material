import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        # initialize weights with random values and add a row of bias
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None  # _ indicates that the variable or method is private, which is only used internally and cannot be called.
        self.input_tensor = None
        self._gradient_weights = None
        self.error_tensor = None

    def forward(self, input_tensor):
        # input_sensor:(batch_size,input_size+1) every row indicates a batch,every column indicates a feature
        batch_size = input_tensor.shape[0]
        self.input_tensor = np.concatenate((input_tensor, np.ones((batch_size, 1))), axis=1)  # add a column of ones
        # weights:(input_size + 1, output_size) actually the size of features
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        # E_n-1 = E_n * W.T and delete the last column
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        self.error_tensor = np.delete(self.error_tensor, -1, axis=1)
        # W_t+1 = W - learning rate * E_n * X.T
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            updated_weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.weights = updated_weights
        return self.error_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
