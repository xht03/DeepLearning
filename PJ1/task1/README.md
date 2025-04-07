# 任务1

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