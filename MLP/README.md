# Project 1 实验报告

## 技术路线

### 隐藏层

多层感知机在单层神经⽹络的基础上引⼊了⼀到多个**隐藏层（hidden layer）**。隐藏层位于输⼊层和输出层之间。

下图展示了⼀个多层感知机的神经⽹络图。输⼊和输出个数分别为 4 和 3，中间的隐藏层中包含了 5 个隐藏单元（hidden unit）。由于输⼊层不涉及计算，图中的多层感知机的层数为2。

在多层感知机中，隐藏层中的神经元和输⼊层中各个输⼊**完全连接**，输出层中的神经元和隐藏层中的各个神经元也**完全连接**。因此，多层感知机中的隐藏层和输出层都是**全连接层**。

![](https://ref.xht03.online/202503162001858.png)

我们以单隐藏层的多层感知机为例，其数学公式如下：

$$
\begin{aligned}
\mathbf{H} & = \phi(\mathbf{W}_1 \mathbf{X} + \mathbf{b}_1) \\
\mathbf{O} & = \mathbf{W}_2 \mathbf{H} + \mathbf{b}_2
\end{aligned}
$$

其中 $\phi(x)$ 是**非线性激活函数**（如 ReLU、Sigmoid 或 Tanh），作用在每个元素上。

---

### 激活函数

在隐藏层中，我们需要引入**激活函数（activation function）**。

这是因为：线性变换只是对数据做仿射变换，如果没有激活函数，多层感知机的每一层输出将是上层输入的线性变换。这样，无论神经网络有多少层，其效果都等价于单层神经网络。

所以，我们需要引入非线性变换，使得神经网络可以拟合任意函数。常用的激活函数有 ReLU、Sigmoid 和 Tanh 等。

#### ReLU 函数

$$
\begin{aligned}
\text{ReLU}(x) & = \max(x, 0) \\
\text{ReLU}'(x) & =
\begin{cases}
1, & x > 0 \\
0, & x \leq 0
\end{cases}
\end{aligned}
$$

#### Sigmoid 函数

$$
\begin{aligned}
\text{sigmoid}(x) & = \frac{1}{1 + \exp(-x)} \\
\text{sigmoid}'(x) & = \text{sigmoid}(x) \cdot(1 - \text{sigmoid}(x))
\end{aligned}
$$

---

### 正向传播

正向传播是指对神经⽹络沿着从输⼊层到输出层的顺序，依次计算并存储模型的中间变量（包括输出）。

python 代码如下：

```python
# 单层前向传播
def single_forward(x, w, b, activation="sigmoid"):
    # 选择激活函数
    if activation == "id":
        activation_func = id
    elif activation == "sigmoid":
        activation_func = sigmod
    elif activation == "relu":
        activation_func = relu
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception("Non-supported activation function")

    # 计算 z 和 o
    z = np.dot(w, x) + b
    o = activation_func(z)
    return z, o
```

```python
# 多层前向传播
def forward(x, params, architecture):
    memory = {} # 存储中间变量
    i = x

    for idx, layer in enumerate(architecture):
        activation = layer["activation"]
        w = params["w" + str(idx + 1)]
        b = params["b" + str(idx + 1)]
        z, o = single_forward(i, w, b, activation)

        memory["i" + str(idx + 1)] = i
        memory["z" + str(idx + 1)] = z
        memory["o" + str(idx + 1)] = o

        i = o

    return i, memory
```

![](https://ref.xht03.online/202503162107738.png)

---

### 反向传播

反向传播指的是计算神经⽹络参数梯度的⽅法。总的来说，反向传播依据微积分中的**链式法则**，沿着从输出层到输⼊层的顺序，依次计算并存储⽬标函数有关神经⽹络各层的中间变量以及参数的梯度。

当我们知道各层参数的梯度后，我们就可以使⽤**小批量随机梯度下降法**来更新这些参数，从而最小化损失函数。

对于每一层反向传播，我们需要计算该层的梯度 $\frac{\partial L}{\partial w}$ 和 $\frac{\partial L}{\partial b}$，以及来自前面一层的梯度 $\frac{\partial L}{\partial x}$。

第 i 层梯度的计算公式公式如下：

$$
\begin{aligned}
\frac{\partial L}{\partial w_i} & = \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_i} \quad \text{（由于第 i 层的输出是第 i+1 层的输入）} \\ \\
& = \frac{\partial L}{\partial x_{i+1}} \cdot \frac{\partial o_i}{\partial z_i} \cdot \mathbf{x}_{i}^T
\end{aligned}
$$

注意到：$\frac{\partial L}{\partial x_{i+1}}$ 是后面一层的梯度，$\frac{\partial o_i}{\partial z_i}$ 是激活函数的导数，$\mathbf{x}_{i}^T$ 是输入的转置。

类似的有：

$$
\begin{aligned}
\frac{\partial L}{\partial b_i} & = \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_i} \\ \\
\frac{\partial L}{\partial x_i} & = \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_i}
\end{aligned}
$$

python 代码如下：

```python
# 单层反向传播
# d: 来自后一层的梯度
# w: 来自当前层的权重
# b: 来自当前层的偏置
# o: 当前层的激活值
# z: 当前层的输出
# i: 当前层的输入
# activation: 当前层的激活函数
def single_backward(d, w, b, o, z, i, activation="sigmoid"):
    m = i.shape[1]  # 样本数

    # 选择激活函数对应的导数
    if activation == "id":
        derivative_func = id_derivative
    elif activation == "sigmoid":
        derivative_func = sigmoid_derivative
    elif activation == "relu":
        derivative_func = relu_derivative
    elif activation == "softmax":
        derivative_func = softmax_derivative
    else:
        raise Exception("Non-supported activation function")

    # 链式求导
    dz = d * derivative_func(z)
    dw = 1 / m * np.dot(dz, i.T)
    db = 1 / m * np.sum(dz, axis=1, keepdims=True)
    di = np.dot(w.T, dz)

    return di, dw, db
```

---

## 代码实现

### 目录结构

- `Func.py`中定义了各种激活函数及其导数。
- `MLP.py`中实现了多层感知机。
- `sin.py`对正弦函数进行了拟合，并评估了拟合效果。
- `digits.py`完成了对手写数字体的识别，并计算了准确率。

[详见代码仓库](https://github.com/xht03/DeepLearning)

---

### 最终效果

**拟合正弦函数**

![](https://ref.xht03.online/202503162137700.png)

![](https://ref.xht03.online/202503162137546.png)

![](https://ref.xht03.online/202503162138583.png)

**识别手写数字**

![](https://ref.xht03.online/202503162138686.png)

![](https://ref.xht03.online/202503162138127.png)