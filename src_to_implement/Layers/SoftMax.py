from Layers.Base import BaseLayer
import numpy as np
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Calculate the exponential value of the input tensor, subtracting the maximum value to improve numerical stability
        # axis=1 表示对每一行（即每个样本）进行操作。
        # keepdims=True 表示保持原有的维度，使得结果的形状与原输入张量的形状兼容。
        # softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z))) for j in range(len(z))
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, error_tensor):
        return self.output * (error_tensor - np.sum(error_tensor * self.output, axis=1, keepdims=True))