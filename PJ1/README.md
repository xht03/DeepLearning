# Project 1

## 目录结构

- data: 数据集

- docs: 实验文档

- lib: 自己实现的深度学习库

- model: 模型参数文件

- task*: 不同实验任务的代码

- test: 测试数据

```bash
.
└── PJ1
    ├── README.md
    ├── data
    ├── docs
    ├── lib
    ├── model
    ├── task1
    ├── task2
    ├── task3
    └── test
```

## 组件库

在 `lib/` 中，我实现了深度学习组件库。

```bash
.
├── Func.py
├── MLP.py
├── Net.py
└── __init__.py
```

`MLP.py` 实现了多层感知机模型。

`Func.py` 实现了激活函数和损失函数，以及必要的数据预处理函数。最初的 task1 由这两个库实现，现保存在 task1 分支中。后续为了兼容卷积层的实现，`Func.py` 进行了重构，现在的 `Func.py` 不适用于 `MLP.py`。

`Net.py` 在 `MLP.py` 的基础上进行了抽象，将每一层网络抽象为一个个类，便于灵活设计网络结构。`Net.py` 还实现了卷积层和池化层，便于实现卷积神经网络。

经过验证，卷积层能正确地实现前向传播和反向传播，且能正确地计算梯度。但是由于缺乏 `BatchNorm` 等正则化手段，模型的训练效果不佳，无法收敛。

使用 `lib/` 库，自定义网络架构格式如下：

```python
# 定义网络架构
architecture = [
    {"module": Flatten},
    {"module": Mlp, "params": {"input_dim": 784, "output_dim": 256, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 256, "output_dim": 64, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 64, "output_dim": 10, "activation": "softmax"}},
]
# 创建网络实例
net = Net(architecture)
```

## 任务

共三个任务，每个任务的代码在 `task*` 目录下。具体任务文档在 `docs/` 目录下。

- 任务1：拟合 $sin(x)$ 函数；`MNIST` 数据集分类

- 任务2：使用 `MNIST` 和 `CIFAR-10` 数据集训练卷积神经网络

- 任务3：使用 `MNIST` 和 `CIFAR-10` 数据集训练残差神经网络，并进行调参实验

### 任务1

任务 1 的核心在于 MLP 的正向传播和反向传播。

![](https://ref.xht03.online/202503162107738.png)

正向传播的数学公式如下：

$$
\begin{aligned}
\mathbf{H} & = \phi(\mathbf{W}_1 \mathbf{X} + \mathbf{b}_1) \\
\mathbf{O} & = \mathbf{W}_2 \mathbf{H} + \mathbf{b}_2
\end{aligned}
$$

反向传播本质上是链式法则的应用。我们只需考虑清楚每一层的链式法则，组合在一起后就能得到完整的反向传播公式。

$$
\begin{aligned}
\frac{\partial L}{\partial w_i} & = \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_i} \quad \text{（由于第 i 层的输出是第 i+1 层的输入）} \\ \\
& = \frac{\partial L}{\partial x_{i+1}} \cdot \frac{\partial o_i}{\partial z_i} \cdot \mathbf{x}_{i}^T
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial b_i} & = \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_i} \\ \\
\frac{\partial L}{\partial x_i} & = \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_i}
\end{aligned}
$$

需要注意的是，任务 1 中的输入输出我都认为是列向量，而非行向量。也就是说，对于 `batch_size` 个输入 `X` ，我们认为它的形状是 `(batch_size, input_featrue)`，而不是 `(input_featrue, batch_size)`。

调参规律我在任务三种会有详细探讨，这里只简单阐述比较容易发现的规律：

- 学习率过大，模型无法收敛，损失函数反复震荡。但是如果过小，会训不动模型。这种情况在卷积层中最为常见。

- 增加全连接层的神经元个数，比增加深度，效果往往会更好。

![正弦函数拟合情况](https://ref.xht03.online/202504071426604.png)

### 任务2

任务 2 的核心在于卷积层的正向传播和反向传播。

其数学原理在[这篇文章](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)中有很清晰的推导。

`Net.py` 中实现了卷积层、池化层的正向传播和反向传播。

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

### 任务3

#### 超参数调优

我对 task1 中的 MNIST 任务进行超参数调优，基准模型如下：

| 层数 | 架构                                                   | 
|------|-------------------------------------------------------|
|  0   | Flatten                                               |
|  1   | Mlp(input_dim=784, output_dim=256, activation="relu") |
|  2   | Dropout(p=0.3)                                        |
|  3   | Mlp(input_dim=256, output_dim=64, activation="relu")  |
|  4   | Dropout(p=0.3)                                        |
|  5   | Mlp(input_dim=64, output_dim=16, activation="softmax")|
|  6   | Dropout(p=0.3)                                        |
|  7   | Mlp(input_dim=16, output_dim=10, activation="softmax")|

基准参数如下：

|     参数名     |      参数值      |
|---------------|------------------|
| loss_function | CrossEntropyLoss |
| activation    | relu             |
| batch_size    | 512              |
| learning_rate | 1                |

然后我们对不同参数，不同网络结构的模型进行了训练，测试其准确率：

| 网络结构 | 激活函数 | 损失函数 | 准确率90%迭代次数 | 最佳准确率 |
|----------|----------|------------|------------------|------------|
| 2 层      | relu    | MSE         | 1000             | 0.88       |


