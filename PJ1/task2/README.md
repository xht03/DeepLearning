# 任务2

任务 2 的核心在于卷积层的正向传播和反向传播。

其数学原理在[这篇文章](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)中有很清晰的推导。

卷积层的正向传播公式如下：

$$ O = conv(X, W) + b $$

反向传播公式如下：

```
dL = dL/dO                    (batch_size, out_channels, height_out, width_out)
dZ = dL/dZ = dL/dO * dO/dZ    (batch_size, out_channels, height_out, width_out)
dW = dL/dW = conv(X, dZ)      (out_channels, in_channels, kernel_size, kernel_size)
db = dL/db = sum(dZ)          (out_channels)
dX = dL/dX = conv(dZ^p, W^r)  (batch_size, in_channels, height, width)
```

## 手写 CNN

在 `Net.py` 中，我自己实现了卷积层、池化层的正向传播和反向传播。`MNIST_.ipynb` 是我自己实现的 MNIST 任务的卷积神经网络。

但是由于缺乏还没有实现 `BatchNorm` 层，所以在训练时，模型的训练效果并不理想，正确率最多达到 20% 左右。

## PyTorch 实现 CNN

在 MNIST 任务上，我们采用如下网络结构：

```python
class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
    
    pass    # 省略
```

在 CIFAR-10 任务上，我们采用如下网络结构：

```python
class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    pass   # 省略
```

MNIST 任务的训练效果是明显的，模型在训练集和验证集上的准确率都能达到 98% 以上。

CIFAR-10 任务的训练效果就差强人意了，模型在验证集上的准确率最高达到 86% 左右。而且是否加入 `BatchNorm` 层对模型的训练效果影响很大，加入 `BatchNorm` 层后，模型的训练效果明显提升。但加入 `Dropout` 层后，效果并不明显，只有 `p=0,3` 时，模型的训练效果才会有所提升。`p=0.5` 时，模型的训练效果反而下降了。