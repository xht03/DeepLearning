import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

# 激活函数及其导数
def id(X):
    return X

def id_derivative(X, dL):
    return dL

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(X, dL):
    return dL * sigmoid(X) * (1 - sigmoid(X))

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X, dL):
    return dL * np.where(X > 0, 1, 0)

def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

def softmax_derivative(X, dL):
    X = softmax(X)
    grad_sum = np.sum(X * dL, axis=-1, keepdims=True)
    return X * (dL - grad_sum)


def gelu(X):
    return 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)))

def gelu_derivative(X, dL):
    return dL * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X ** 3))) + \
           0.5 * X * (1 - np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X ** 3)) ** 2) * \
           (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * X ** 2))

# --------------------------

# 损失函数
def square_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def square_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


# y_true: (batch_size, num_classes)
def cross_entropy_loss(y_true, y_pred):
    # 防止 y_pred 中出现 0 或 1
    epsilon = 1e-2
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_derivative(y_true, y_pred):
    # 防止 y_pred 中出现 0 或 1
    epsilon = 1e-2
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (-y_true / y_pred) / y_true.shape[0]

# --------------------------

def one_hot(labels, num_classes):
    # one-hot 编码
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def shuffle(x, y):
    np.random.seed(0)
    indices = np.random.permutation(len(x))
    return x[indices], y[indices]