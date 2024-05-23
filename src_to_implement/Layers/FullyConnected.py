import numpy as np

from Layers.Base import BaseLayer
class FullyConnected(BaseLayer):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0,1)
        self.input_size = input_size
        self.output_size = output_size


    def forward(self, input_tensor):
