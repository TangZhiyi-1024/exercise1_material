import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.label_tensor = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # loss = sum of batch(-In(y_h+eps)) when y=1
        eps = np.finfo(float).eps
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        loss = -np.sum(np.log(self.prediction_tensor[label_tensor == 1]+eps))
        return loss

    def backward(self, label_tensor):
        # e_n = -y/(y_h + eps)
        eps = np.finfo(float).eps
        return -label_tensor / (self.prediction_tensor + eps)
