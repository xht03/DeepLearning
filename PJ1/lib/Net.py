import numpy as np
import os
import matplotlib.pyplot as plt
from lib.Func import *


# architecture = [
#     {"module": Flatten},
#     {"module": Mlp, "param": {"in_features": 784, "out_features": 256}},
#     {"module": Gelu},
#     {"module": Dropout, "param": {"p": 0.2}},
#     {"module": Mlp, "param": {"in_features": 256, "out_features": 64}},
#     {"module": Gelu},
#     {"module": Dropout, "param": {"p": 0.2}},
#     {"module": Mlp, "param": {"in_features": 64, "out_features": 16}},
#     {"module": Gelu},
#     {"module": Dropout, "param": {"p": 0.2}},
#     {"module": Mlp, "param": {"in_features": 16, "out_features": 10}},
#     {"module": Softmax},
# ]


class Net:
    def __self__(self, architecture):
        self.architecture = architecture
        self.net = []
        for layer in architecture:
            self.net.append(layer["module"](**layer.get("params", {})))

    # def save(self, path):


class Mlp:
    def __init__(self, input_dim, output_dim, W=None, b=None, activation="sigmoid"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.X = None
        self.Z = None
        self.O = None

        if W is not None:
            self.W = W
            self.b = b
        else:
            self.W = np.random.randn(output_dim, input_dim) * 0.01
            self.b = np.random.randn(output_dim, 1) * 0.01

        if activation == "id":
            self.activation = id
            self.derivative = id_derivative
        if activation == "sigmoid":
            self.activation = sigmoid
            self.derivative = sigmoid_derivative
        elif activation == "relu":
            self.activation = relu
            self.derivative = relu_derivative
        elif activation == "softmax":
            self.activation = softmax
            self.derivative = softmax_derivative
        else:
            raise Exception("Non-supported activation function")

    def get_params(self):
        return self.W, self.b

    # X: (batch_size, input_dim)
    # W: (output_dim, input_dim)
    # b: (output_dim)
    # Z: (batch_size, output_dim)
    # O: (batch_size, output_dim)
    def forward(self, X):
        assert X.shape[1] == self.input_dim
        Z = np.dot(X, self.W.T) + self.b
        O = self.activation(Z)

        self.X = X
        self.Z = Z
        self.O = O

        return O

    # dL = dL/dO                        (batch_size, output_dim)
    # dZ = dL/dZ                        (batch_size, output_dim)
    # dW = dL/dW = dL/dZ * dZ/dW        (output_dim, input_dim)
    # db = dL/db                        (output_dim)
    # dX = dL/dX = dL/dZ * dZ/dX        (batch_size, input_dim)
    def backward(self, dL, lr):
        m = self.X.shape[0]  # 样本数
        dZ = dL * self.derivative(self.Z)
        dW = np.dot(dZ.T, self.X) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W)

        self.W -= lr * dW
        self.b -= lr * db

        return dX
    

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

        self.X = None
        self.O = None

    def forward(self, X):
        self.X = X
        # Create a mask with the same shape as X, where each element is 0 with probability p and 1 with probability (1-p)
        self.mask = np.random.rand(*X.shape) > self.p
        # Apply the mask to X
        self.O = X * self.mask / (1 - self.p)
        return self.O
        

    def backward(self, dL):
        return dL * self.mask / (1 - self.p)


class Flatten:
    # X: (batch_size, height, width)
    # O: (batch_size, height * width)
    def __init__(self):
        self.X = None
        self.O = None
        self.shape = None

    def forward(self, X):
        self.X = X
        self.shape = X.shape
        self.O = X.reshape(X.shape[0], -1)
        return self.O

    def backward(self, dL, lr):
        # Reshape dL back to the original shape
        dX = dL.reshape(self.shape)
        return dX


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, W=None, b=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if W is not None:
            self.W = W
            self.b = b
        else:
            self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
            self.b = np.random.randn(out_channels) * 0.01

        self.X = None
        self.O = None


    def get_params(self):
        return self.W, self.b

    # O = conv(X, W) + b
    # X: (batch_size, in_channels, height, width)
    # W: (out_channels, in_channels, kernel_size, kernel_size)
    # b: (out_channels, 1)
    # O: (batch_size, out_channels, height_out, width_out)
    # height_out = (height - kernel_size + 2 * padding) / stride + 1
    # width_out = (width - kernel_size + 2 * padding) / stride + 1
    def forward(self, X):
        self.X = X
        batch_size, in_channels, height, width = X.shape

        # 计算 O 的高度和宽度
        height_out = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        width_out = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # 初始化输出矩阵 O
        O = np.zeros((batch_size, self.out_channels, height_out, width_out))

        # 填入 padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
            X_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width] = X
        else:
            X_padded = X

        # 计算卷积
        for i in range(batch_size):
            for j in range(self.out_channels):
                for h in range(height_out):
                    for w in range(width_out):
                        # 计算 X 对应的区域
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # 提取 X 对应的区域
                        region = X_padded[i, :, h_start:h_end, w_start:w_end]

                        # region: (in_channels, kernel_size, kernel_size)
                        # W[j]:   (in_channels, kernel_size, kernel_size)
                        O[i, j, h, w] = np.sum(region * self.W[j]) + self.b[j]

        self.O = O
        return O

    # O = conv(X, W) + b
    # dL = dL/dO                    (batch_size, out_channels, height_out, width_out)
    # dW = dL/dW = conv(X, dL)      (out_channels, in_channels, kernel_size, kernel_size)
    # db = dL/db = sum(dL)          (out_channels)
    # dX = dL/dX = conv(dL^p, W^r)  (batch_size, in_channels, height, width)
    def backward(self, dL, lr):
        batch_size, in_channels, height, width = self.X.shape

        # 填入 padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
            X_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width] = self.X
        else:
            X_padded = self.X
        
        dX_padded = np.zeros_like(X_padded)
        dX = np.zeros_like(self.X)

        # --------------------------------------
        dW = np.zeros_like(self.W)

        for i in range(batch_size):
            for j in range(self.out_channels):
                for h in range(dL.shape[2]):
                    for w in range(dL.shape[3]):
                        # 计算 X_padded 对应的区域
                        h_start = h * self.stride
                        w_start = w * self.stride
                        # (in_channels, kernel_size, kernel_size) = (in_channels, kernel_size, kernel_size) * scalar
                        dW[j] += X_padded[i, :, h_start : h_start + self.kernel_size, w_start : w_start + self.kernel_size] * dL[i, j, h, w]

        # ---------------------------------------
        db = np.sum(dL, axis=(0, 2, 3)) / batch_size

        # ---------------------------------------
        for i in range(batch_size):
            for j in range(self.out_channels):
                for h in range(dL.shape[2]):
                    for w in range(dL.shape[3]):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        dX_padded[i, :, h_start : h_start + self.kernel_size, w_start : w_start + self.kernel_size] += self.W[j] * dL[i, j, h, w]

        # 提取 dX
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding : self.padding + self.X.shape[2], self.padding : self.padding + self.X.shape[3]]
        else:
            dX = dX_padded

        # --------------------------------------
        self.W -= lr * dW
        self.b -= lr * db

        return dX
    

class MaxPool2d:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.X = None
        self.O = None
        self.mask = None

    # X: (batch_size, channels, height, width)
    def forward(self, X):
        def forward(self, X):
            self.X = X
            batch_size, channels, height, width = X.shape
            
            # 计算 O 的高度和宽度
            height_out = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
            width_out = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
            
            # 填入 padding
            if self.padding > 0:
                X_padded = np.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))
                X_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width] = X
            else:
                X_padded = X
            
            # 初始化输出矩阵 O
            O = np.zeros((batch_size, channels, height_out, width_out))
            # 初始化 mask (记录最大值位置)
            self.mask = np.zeros_like(X_padded)
            
            # 最大池化
            for i in range(batch_size):
                for j in range(channels):
                    for h in range(height_out):
                        for w in range(width_out):
                            # 计算 X_padded 对应的区域
                            h_start = h * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size
                            
                            window = X_padded[i, j, h_start:h_end, w_start:w_end]
                            max_val = np.max(window)
                            O[i, j, h, w] = max_val
                            
                            # 记录最大值的位置
                            max_pos = np.unravel_index(np.argmax(window), window.shape)
                            self.mask[i, j, h_start + max_pos[0], w_start + max_pos[1]] = 1
            
            self.O = O
            return O
        
    # dL: (batch_size, channels, height_out, width_out)
    def backward(self, dL, lr):
        batch_size, channels, height, width = self.X.shape
        
        # 填入 padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))
            X_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width] = self.X
        else:
            X_padded = self.X
        
        dX_padded = np.zeros_like(X_padded)
        
        # 最大池化的反向传播
        for i in range(batch_size):
            for j in range(channels):
                for h in range(dL.shape[2]):
                    for w in range(dL.shape[3]):
                        # 计算 X_padded 对应的区域
                        h_start = h * self.stride
                        w_start = w * self.stride
                        # 将 dL 的值传递到最大值的位置
                        dX_padded[i, j, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] += dL[i, j, h, w] * self.mask[i, j, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
        
        # 提取 dX
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width]
        else:
            dX = dX_padded
        
        return dX
