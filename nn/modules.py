"""
模型定义
"""
from typing import List


class BaseModule(object):
    def __init__(self, name=''):
        self.name = name
        self.weights = dict()
        self.gradients = dict()
        self.features = None

    def forward(self, x):
        pass

    def backward(self, in_gradient):
        pass

    def update_gradient(self, lr):
        pass

    def load_weights(self, weights):
        for key in self.weights.keys():
            self.weights[key] = weights[key]
    

class Model(BaseModule):
    def __init__(self, layers: List[BaseModule], **kwargs):
        super(Model, self).__init__(**kwargs)
        self.layers = layers
        #收集所有权重和梯度
        for l in self.layers:
            self.weights.update(l.weights)
            self.gradients.update(l.gradients)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, in_gradient):
        for l in self.layers[::-1]:
            in_gradient = l.backward(in_gradient)

    def update_gradient(self, lr):
        for l in self.layers:
            l.update_gradient(lr)

    def load_weights(self, weights):
        for l in self.layers:
            l.load_weights(weights)