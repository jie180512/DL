"""
定义损失函数
"""
import numpy as np


def mean_squared_loss(y_predict, y_true):
    """
    均方误差损失函数
    """
    loss = np.mean(np.sum(np.square(y_predict - y_true), axis=-1))
    dy = y_predict - y_true
    return loss, dy

def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵损失函数
    """
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true*np.log(y_probability), axis=-1))
    dy = y_probability - y_true
    return loss, dy