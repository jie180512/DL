"""
模型定义
"""
from typing import List


class BaseModule(object):
    def __init__(self, name=''):
        """

        :param name: 层名
        """
        self.name = name
        self.weights = dict()  # 权重参数字典
        self.gradients = dict()  # 梯度字典
        self.in_features = None  # 输入的feature map

    def forward(self, x):
        pass

    def backward(self, in_gradient):
        pass

    def update_gradient(self, lr):
        pass

    def load_weights(self, weights):
        """
        加载权重
        :param weights:
        :return:
        """
        for key in self.weights.keys():
            self.weights[key] = weights[key]


class Model(BaseModule):
    """
    网络模型
    """

    def __init__(self, layers: List[BaseModule], **kwargs):
        super(Model, self).__init__(**kwargs)
        self.layers = layers
        # 收集所有权重和梯度
        for l in self.layers:
            self.weights.update(l.weights)
            self.gradients.update(l.gradients)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
            # print('forward layer:{},feature:{}'.format(l.name, np.max(x)))
        # 网络结果返回
        return x

    def backward(self, in_gradient):
        # 反向传播
        for l in self.layers[::-1]:
            in_gradient = l.backward(in_gradient)
            # print('backward layer:{},gradient:{}'.format(l.name, np.max(in_gradient)))

    def update_gradient(self, lr):
        for l in self.layers:
            l.update_gradient(lr)

    def load_weights(self, weights):
        """
        加载模型权重
        :param weights:
        :return:
        """
        # 逐层加载权重
        for l in self.layers:
            l.load_weights(weights)