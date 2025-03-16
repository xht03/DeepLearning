import numpy as np
import matplotlib.pyplot as plt

# 激活函数及其导数
def id(x):
    return x

def id_derivative(x):
    return 1

def sigmod(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmod(x) * (1 - sigmod(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

# 损失函数
def square_loss(y_true, y_pred):
    return np.mean((1 / 2) * (y_true - y_pred) ** 2)

def square_derivative(y_true, y_pred):
    return y_pred - y_true

def cross_entropy_loss(y_true, y_pred):
    m = y_pred.shape[1]  # 样本数
    loss = -1 / m * (np.dot(y_true, np.log(y_pred).T) + np.dot(1 - y_true, np.log(1 - y_pred).T))
    return np.squeeze(loss)

def cross_entropy_derivative(y_true, y_pred):
    return -(np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))