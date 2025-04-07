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



