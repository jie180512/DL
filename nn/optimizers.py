"""
优化器
"""
import numpy as np
from modules import Model


def _copy_weights_to_zeros(weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result

class SGD(object):
    def __init__(self, weights, lr=0.01, momentum=0.9, decay=1e-5):
        self.v = _copy_weights_to_zeros(weights) #累积动量大小
        self.iterations = 0
        self.lr = self.init_lr = lr
        self.momentum = momentum
        self.decay = decay

    def iterate(self, m: Model):
        #更新学习率
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        #更新动量和梯度
        for layers in m.layers:
            for key in layers.weights.keys():
                self.v[key] = self.momentum * self.v[key] + self.lr * layers.gradients[key]
                layers.weights[key] -= self.v[key]

        self.iterations += 1

class adaGrad(object):
    def __init__(self, weights, lr=0.01, epsilon=1e-6, decay=0):
        self.s = _copy_weights_to_zeros(weights)  # 权重平方和累加量
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.epsilon = epsilon
        self.decay = decay

    def iterate(self, m: Model):
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        # 更新权重平方和累加量 和 梯度
        for layer in m.layers:
            for key in layer.weights.keys():
                self.s[key] += np.square(layer.gradients[key])
                layer.weights[key] -= self.lr * layer.gradients[key] / np.sqrt(self.s[key] + self.epsilon)
        # 更新迭代次数
        self.iterations += 1


class RmsProp(object):
    def __init__(self, weights, gamma=0.9, lr=0.01, epsilon=1e-6, decay=0):
        self.s = _copy_weights_to_zeros(weights)  # 权重平方和累加量
        self.gamma = gamma
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.epsilon = epsilon
        self.decay = decay

    def iterate(self, m: Model):
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        # 更新权重平方和累加量 和 梯度
        for layer in m.layers:
            for key in layer.weights.keys():
                self.s[key] = self.gamma * self.s[key] + (1 - self.gamma) * np.square(layer.gradients[key])
                layer.weights[key] -= self.lr * layer.gradients[key] / np.sqrt(self.s[key] + self.epsilon)
        # 更新迭代次数
        self.iterations += 1