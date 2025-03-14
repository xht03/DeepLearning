import numpy as np
from abc import ABC, abstractmethod

# 激活函数 抽象类
class Activation(ABC):
    @abstractmethod
    def forward(self, U):
        pass

    @abstractmethod
    def backward(self, U):
        pass


class Id(Activation):
    def forward(self, U):
        return U

    def backward(self, U):
        return 1

# ------------------------

class Relu(Activation):
    def forward(self, U):
        return np.maximum(0, U)

    def backward(self, U):
        return np.where(U > 0, 1, 0)


class Sigmoid(Activation):
    def forward(self, U):
        return 1 / (1 + np.exp(-U))

    def backward(self, U):
        X_ = self.forward(U)
        return X_ * (1 - X_)
