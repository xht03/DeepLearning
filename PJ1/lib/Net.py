import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.Func import *



class Net:
    def __init__(self, architecture):
        self.architecture = architecture
        self.net = []
        for layer in architecture:
            self.net.append(layer["module"](**layer.get("params", {})))


    def save_params(self, filepath):
        params_dict = {}
        
        for i, layer in enumerate(self.net):
            if hasattr(layer, 'get_params'):
                # Get parameters
                params = layer.get_params()
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(params, tuple):
                    params_dict[str(i)] = [param.tolist() if isinstance(param, np.ndarray) else param for param in params]
                else:
                    params_dict[str(i)] = params.tolist() if isinstance(params, np.ndarray) else params
        
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存参数到 JSON 文件
        with open(filepath, 'w') as f:
            json.dump(params_dict, f)
        
        print(f"Model parameters saved to {filepath}")


    def load_params(self, filepath):
        # 从 JSON 文件加载参数
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        
        # 设置参数
        for i_str, params in params_dict.items():
            i = int(i_str)
            if i < len(self.net) and hasattr(self.net[i], 'get_params'):
                if isinstance(self.net[i], Mlp):
                    # Convert lists back to numpy arrays
                    W = np.array(params[0])
                    b = np.array(params[1])
                    self.net[i].W, self.net[i].b = W, b
                elif isinstance(self.net[i], Conv2d):
                    # Convert lists back to numpy arrays
                    W = np.array(params[0])
                    b = np.array(params[1])
                    self.net[i].W, self.net[i].b = W, b
        
        print(f"Model parameters loaded from {filepath}")
    

    # X: (batch_size, input_dim)
    def forward(self, X):
        outputs = [X]
        for layer in self.net:
            X = layer.forward(X)
            outputs.append(X)
        return outputs
    
    def backward(self, outputs, dL, lr):
        for i in range(len(self.net) - 1, -1, -1):
            dL = self.net[i].backward(outputs[i], dL, lr)

    def train(self, X, Y, batch_size, lr, lossfunc="cross_entropy"):
        # 损失函数
        if lossfunc == "cross_entropy":
            self.loss = cross_entropy_loss
            self.loss_derivative = cross_entropy_derivative
        elif lossfunc == "square":
            self.loss = square_loss
            self.loss_derivative = square_derivative
        else:
            raise Exception("Non-supported loss function")
        
        # 训练
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            Y_batch = Y[i:i + batch_size]

            # 前向传播
            outputs = self.forward(X_batch)

            # 计算损失
            y_hat = outputs[-1]
            
            # 反向传播
            dL = self.loss_derivative(Y_batch, y_hat)
            self.backward(outputs, dL, lr)

            print(f"Batch {i // batch_size + 1}/{len(X) // batch_size}, Loss: {self.loss(Y_batch, y_hat):.4f}")
        
    def predict(self, X):
        for layer in self.net:
            if isinstance(layer, Dropout):
                continue
            X = layer.forward(X)
        return X


class Mlp:
    def __init__(self, input_dim, output_dim, W=None, b=None, activation="sigmoid"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if W is not None:
            self.W = W
            self.b = b
        else:
            self.W = np.random.randn(output_dim, input_dim) * 0.01
            self.b = np.random.randn(output_dim) * 0.01

        if activation == "id":
            self.activation = id
            self.derivative = id_derivative
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.derivative = sigmoid_derivative
        elif activation == "relu":
            self.activation = relu
            self.derivative = relu_derivative
        elif activation == "softmax":
            self.activation = softmax
            self.derivative = softmax_derivative
        elif activation == "gelu":
            self.activation = gelu
            self.derivative = gelu_derivative
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
        return O

    # dL = dL/dO                        (batch_size, output_dim)
    # dZ = dL/dZ = dL/dO * dO/dZ        (batch_size, output_dim)
    # dW = dL/dW = dL/dZ * dZ/dW        (output_dim, input_dim)
    # db = dL/db = dL/dZ * dZ/db = dZ   (output_dim)
    # dX = dL/dX = dL/dZ * dZ/dX        (batch_size, input_dim)
    def backward(self, X, dL, lr):
        batch_size = X.shape[0]  # 样本数
        assert X.shape == (batch_size, self.input_dim)
        assert dL.shape == (batch_size, self.output_dim)

        Z = np.dot(X, self.W.T) + self.b
        dZ = self.derivative(Z, dL)
        dW = np.dot(dZ.T, X) / batch_size
        db = np.sum(dZ, axis=0) / batch_size
        dX = np.dot(dZ, self.W)

        self.W -= lr * dW
        self.b -= lr * db

        return dX
    

class Dropout:
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, X):
        self.mask = np.random.rand(*X.shape) > self.p
        return X * self.mask / (1 - self.p)
        
    def backward(self, X, dL, lr):
        return dL * self.mask / (1 - self.p)
    

class Flatten:
    # X: (batch_size, channels, height, width)
    # O: (batch_size, channels * height * width)
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, X, dL, lr):
        # Reshape dL back to the original shape
        return dL.reshape(self.shape)


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

    # O = conv(X, W) + b
    # X: (batch_size, in_channels, height, width)
    # W: (out_channels, in_channels, kernel_size, kernel_size)
    # b: (out_channels, 1)
    # O: (batch_size, out_channels, height_out, width_out)
    # height_out = (height - kernel_size + 2 * padding) / stride + 1
    # width_out = (width - kernel_size + 2 * padding) / stride + 1
    def forward(self, X):
        batch_size, in_channels, height, width = X.shape

        # 计算 O 的高度和宽度
        height_out = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        width_out = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # 填入 padding
        # 填入 padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
            X_padded[:, :, self.padding:self.padding + height, self.padding:self.padding + width] = X
        else:
            X_padded = X

        # 初始化输出矩阵 O
        O = np.zeros((batch_size, self.out_channels, height_out, width_out))

        # 计算卷积
        # O = conv(X, W) + b
        for h in range(height_out):
            for w in range(width_out):
                # 计算 X_padded 对应的区域
                h_start = h * self.stride
                h_end = h_start + self.kernel_size
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                X_region = X_padded[:, :, h_start:h_end, w_start:w_end]

                # 卷积操作
                for k in range(self.out_channels):
                    O[:, k, h, w] = np.sum(X_region * self.W[k], axis=(1, 2, 3))

        return O + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    # O = conv(X, W) + b
    # dL = dL/dO                    (batch_size, out_channels, height_out, width_out)
    # dW = dL/dW = conv(X, dL)      (out_channels, in_channels, kernel_size, kernel_size)
    # db = dL/db = sum(dL)          (out_channels)
    # dX = dL/dX = conv(dL^p, W^r)  (batch_size, in_channels, height, width)
    def backward(self, X, dL, lr):

        X = X.astype(np.float64)
        dL = dL.astype(np.float64)

        batch_size, in_channels, height, width = X.shape
        assert in_channels == self.in_channels

        # 填入 padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, in_channels, height + 2 * self.padding, width + 2 * self.padding))
            X_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width] = X
        else:
            X_padded = X
        
        dX_padded = np.zeros_like(X_padded)
        dX = np.zeros_like(X)

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
            dX = dX_padded[:, :, self.padding : self.padding + X.shape[2], self.padding : self.padding + X.shape[3]]
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

    # X: (batch_size, channels, height, width)
    def forward(self, X):
        def forward(self, X):
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
    def backward(self, X, dL, lr):
        batch_size, channels, height, width = X.shape
        
        # 填入 padding
        if self.padding > 0:
            X_padded = np.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))
            X_padded[:, :, self.padding : self.padding + height, self.padding : self.padding + width] = X
        else:
            X_padded = X
        
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
