import numpy as np

from MyFunc import *


class Net:
    def __init__(self, net_info, seed=0):
        np.random.seed(seed)

        self.net = []
        for layer in net_info:
            in_features = layer["in_features"]
            out_features = layer["out_features"]
            activation = layer["activation"]
            W = np.random.randn(out_features, in_features) * 0.1
            b = np.random.randn(out_features) * 0.1
            self.net.append((W, b, activation()))

    # X: (batch_size, in_features)
    # W: (out_features, in_features)
    # b: (out_features)
    # U: (batch_size, out_features)
    def forward(self, X):
        outputs = [{"U": None, "X": X}]
        for layer in self.net:
            W, b, f = layer
            U = np.dot(X, W.T) + b
            X = f.forward(U)
            outputs.append({"U": U, "X": X})
        return outputs

    # U[1] = W[0] * X[0] + b[0], X[1] = f(U[1])
    def backward(self, outputs, dX_, lr=0.01):
        for i in range(len(self.net) - 1, -1, -1):
            # W: (out, in)
            # b: (out)
            W, b, f = self.net[i]

            # U1: (batch, out)
            # X:  (batch, in)
            U1 = outputs[i + 1]["U"]
            X = outputs[i]["X"]

            # 逐元素相乘
            # dX1: (batch, out)
            # dU1: (batch, out)
            # dL/dU1 = dL/dX1 * dX1/dU1
            #        = dL/dX1 * f'(U1)
            dU1_ = dX_ * f.backward(U1)

            batch_size = X.shape[0]

            # dW_: (out, in)
            # dL/dW = dL/dU1 * dU1/dW
            #       = dL/dU1 * X
            dW_ = np.dot(dU1_.T, X) / batch_size

            # db_: (out)
            # dL/db = dL/dU1 * dU1/db
            #       = dL/dU1
            db_ = np.sum(dU1_, axis=0) / batch_size

            # dX: (batch, in)
            # dL/dX = dL/dU1 * dU1/dX
            #       = dL/dU1 * W
            dX_ = np.dot(dU1_, W)

            # 更新参数
            W -= lr * dW_
            b -= lr * db_

    def train(self, X, Y, batch_size, lr):
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            outputs = self.forward(X_batch)

            dX = outputs[-1]["X"] - Y_batch
            self.backward(outputs, dX, lr)

    def pred(self, X):
        return self.forward(X)[-1]["X"]
