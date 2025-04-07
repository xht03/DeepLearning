# 任务3

## BP 超参数调优

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
|----------|----------|--------------|------------------|------------|
| 2 层     | relu     | MSE          | 8                | 97.46%    |
| 2 层     | relu     | CrossEntropy | 4                | 97.66%    |
| 3 层     | relu     | CrossEntropy | 5                | 97.61%    |
| 4 层     | relu     | CrossEntropy | 296              | 34.77%    |
| 2 层     | sigmoid  | CrossEntropy | 14               | 96.46%    |
| 3 层     | sigmoid  | CrossEntropy | 28               | 96.23%    |

所以，

- 分类任务中，`CrossEntropyLoss` 的效果优于 `MSE` 。

- `relu` 激活函数的效果优于 `sigmoid` 激活函数。

- 随着深度增加，训练时间大幅度增加，准确率提升不明显。且可能出现梯度消失的问题。

我们以 3 层基准架构为例，测试不同消融层数的模型的训练效果：

| 网络结构  | 丢弃率  | 准确率90%迭代次数  | 训练20轮后的准确率 |
|----------|--------|------------------|-----------------|
| 3层      | 0      | 13               | 97.15%          |
| 3层      | 0.1    | 10               | 97.46%          |
| 3层      | 0.2    | 11               | 97.47%          |
| 3层      | 0.3    | 9                | 97.33%          |
| 3层      | 0.5    | 9                | 95.74%          |

所以，

- `Dropout` 层的丢弃率过大，模型的训练效果越差。

- 适当的丢弃率能提升模型的训练效果。

| 网络结构 | 批大小 | 初始学习率 | 准确率90%迭代次数 | 最佳准确率 |
|---------|--------|------------|---------------|------------|
| 3 层     | 512    | 1          | 13            | 97.15%     |
| 3 层     | 256    | 1          | 4             | 97.38%     |
| 3 层     | 128    | 1          | 1             | 97.51%     |
| 3 层     | 64     | 1          | 2             | 97.70%     |
| 3 层     | 128    | 0.1        | 2             | 97.66%     |
| 3 层     | 128    | 0.01       | 15            | 97.53%     |

所以，

- 批大小越小，模型准确率越高，学得越快。

- 初始学习率越大，学习速度越快。但学习率越小，模型准确率会更高；但如果太小，则模型准确率反而会下降。

## CNN 超参数调优

我对 task2 中的 CIFAR-10 任务进行超参数调优，基准参数如下：

|     参数名     |      参数值      |
|---------------|------------------|
| loss_function | CrossEntropyLoss |
| activation    | relu             |
| batch_size    | 128              |
| initial_lr    | 0.01             |
| epochs        | 20               |
| lr_patience   | 10               |
| lr_decay      | 0.5              |
| dropout       | 0.3              |
| kernel_size   | 3                |

我们考虑卷积层和池化层数的影响：

| [Conv+Poll]层数 | 参数量 | 最佳准确率 |
|----------------|--------|-----------|
| 1层             | 8.4M  | 71.91%    |
| 2层             | 4.2M  | 77.82%    |
| 3层             | 2.2M  | 81.23%    |
| 4层             | 1.5M  | 78.99%    |

我们再考虑不同卷积核大小的影响：

| [Conv+Poll]层数 | 卷积核 | 参数量 | 最佳准确率    |
|-----------------|--------|--------|-----------|
| 1层             | 3      | 8.4M   | 71.91%    |
| 1层             | 5      | 8.4M   | 72.14%    |
| 1层             | 7      | 8.4M   | 71.44%    |

我们再考虑不同池化层的影响：

| [2Conv+Poll]层数 | 池化方式 | 最佳准确率 |
|-----------------|---------|-----------|
| 1层             | Max     | 75.17%    |
| 1层             | Avg     | 74.29%    |

## ResNet

在之前的调参中，CNN 随着层数增加，训练误差反而上升（并非过拟合，无法通过 Dropout 进行优化）。而且随着层数增加，训练时间大幅度增加，准确率提升不明显。且可能出现梯度消失的问题。

对此，可以使用残差神经网络进行优化。残差神经网络的核心思想是**跳跃连接（Shortcut Connection）**，将每一层的输出与输入进行相加，形成一个残差块。这样可以避免梯度消失的问题，并且可以加速模型的训练。

![](https://ref.xht03.online/202504071619145.png)

具体原理可以参照这篇[文章](https://arxiv.org/abs/1512.03385)。

核心代码如下：

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不匹配，使用1x1卷积进行维度调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = nn.ReLU()(out)
        return out
```