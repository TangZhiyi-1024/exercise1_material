
import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):        #  perform a linear operation
    def __init__(self, input_size, output_size):
        super().__init__()      # call base class constructor
        self.trainable = True
        # initialize weights with random values
        self.weights = np.random.rand(input_size+1, output_size)    # add a row of bias
        self._optimizer = None      # _表示变量或方法是一个私有变量，仅供内部使用，无法被调用
        self.input_tensor = None
        self._gradient_weights = None

    def initialize(self, weights_initializer, bias_initializer):
        pass

    @property
    def optimizer(self):        # call the methods just like properties
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):     # change the optimizer if needed
        self._optimizer = optimizer

    def forward(self, input_tensor):
        input_tensor = np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1) # add a column of ones
        self.input_tensor = input_tensor
        output_tensor = np.dot(input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        # dot product of error and weights transposed
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        self.error_tensor = np.delete(self.error_tensor, -1, axis=1)    # delete the latest column: fake errors besuces of the added column in the input
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

