import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

# 激活函数及其导数
def id(x):
    return x

def id_derivative(x):
    return 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_x_sum = e_x.sum(axis=0)
    e_x_sum = np.where(e_x_sum == 0, 1e-12, e_x_sum)    # 防止溢出
    return e_x / e_x_sum

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) + \
           0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)) ** 2) * \
           (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2))

# --------------------------

# 损失函数
def square_loss(y_true, y_pred):
    return np.mean((1 / 2) * (y_true - y_pred) ** 2)

def square_derivative(y_true, y_pred):
    return y_pred - y_true


def cross_entropy_loss(y_true, y_pred):
    # 防止 y_pred 中出现 0 或 1
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_pred.shape[0]  # 样本数
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return np.squeeze(loss)

def cross_entropy_derivative(y_true, y_pred):
    # 防止 y_pred 中出现 0 或 1
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred

# --------------------------

def one_hot(labels, num_classes):
    # one-hot 编码
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def shuffle(x, y):
    # 获取第一维的大小
    n_samples = x.shape[0]
    # 生成第一维的随机排列
    indices = np.random.permutation(n_samples)
    # 使用高级索引打乱第一维
    return x[indices], y[indices]