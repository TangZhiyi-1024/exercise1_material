import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        pass

    def forward(self, prediction_tensor, label_tensor):
        num_samples = prediction_tensor.shape[0]
        #self.prediction_tensor = np.clip(prediction_tensor, 1e-15, 1 - 1e-15)
        loss = np.sum(label_tensor * -np.log(self.prediction_tensor + np.finfo(float).eps)) / num_samples
        return loss

    def backward(self, label_tensor):
        error_tensor = -(label_tensor / (self.prediction_tensor + np.finfo(float).eps))
        return error_tensor