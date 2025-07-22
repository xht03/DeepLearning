# PJ2 实验报告

## 算法原理

### Viterbi算法

Viterbi算法是一种动态规划方法，用于在HMM和CRF等模型中寻找概率（或得分）最大的标签序列。其递推公式为：

$$
\delta_t(j) = \max_{i} \left[ \delta_{t-1}(i) + \mathbf{T}(i,j) + \mathbf{E}_t(j) \right]
$$

其中，$\delta_t(j)$为到达时刻$t$标签$j$的最大得分，$\mathbf{T}(i,j)$为转移分数，$\mathbf{E}_t(j)$为发射分数。

### HMM参数估计

HMM通过极大似然估计（MLE）进行参数学习，包括初始概率$\pi$、转移概率$A$和发射概率$B$。为避免零概率，采用加一平滑：

- 初始概率：$\hat{\pi}_i = \frac{\text{Count}(y_1 = i) + 1}{\text{总句子数} + \text{标签类别数}}$
- 转移概率：$\hat{a}_{ij} = \frac{\text{Count}(i \to j) + 1}{\text{Count}(i \to *) + \text{标签类别数}}$
- 发射概率：$\hat{b}_i(w) = \frac{\text{Count}(i \text{生成} w) + 1}{\text{Count}(i) + \text{词汇表大小}}$

### CRF前向-后向算法与梯度

CRF通过前向（$\alpha$）和后向（$\beta$）算法高效计算配分函数$Z$和特征的期望计数，实现对数似然的梯度计算：

- 前向递推：$\alpha_t(j) = \sum_{i=1}^N \alpha_{t-1}(i) \cdot \mathbf{T}(i,j) \cdot \mathbf{E}_t(j)$
- 后向递推：$\beta_t(i) = \sum_{j=1}^N \mathbf{T}(i,j) \cdot \mathbf{E}_{t+1}(j) \cdot \beta_{t+1}(j)$
- 配分函数：$Z = \sum_{j=1}^N \alpha_T(j)$
- 权重梯度：$\frac{\partial \log P(y|x)}{\partial w_f} = \text{Count}(f \text{ in true path}) - \mathbb{E}_{P(y|x)}[\text{Count}(f)]$
- 转移矩阵梯度：$\frac{\partial \log P(y|x)}{\partial T(i,j)} = \text{Count}(i \to j \text{ in true path}) - \mathbb{E}_{P(y|x)}[\text{Count}(i \to j)]$

### CRF+Transformer

CRF+Transformer模型将Transformer编码器提取的上下文特征作为CRF的发射分数输入，结合自注意力机制和序列标注的结构约束，提升了模型对长距离依赖的建模能力。

---

## 关键细节

### 数据处理

使用`process_data`函数读取NER数据集，按句子分割，统计标签和词汇表，生成训练和验证集。

### CRF模型实现

- 特征模板：采用窗口特征（如U00:%x[-2,0]等）和bigram特征，见`feature_templates`。
- 前向/后向算法：分别实现了`forward`和`backward`方法，递推计算$\alpha$和$\beta$。
- 维特比解码：`viterbi_decode`方法实现最优路径搜索。
- 梯度计算：`compute_gradients`方法利用前向-后向算法，计算真实路径与期望路径的特征计数差，作为权重和转移矩阵的梯度。
- 训练过程：`train`方法对每个样本累积梯度并更新参数，输出每轮平均损失。

### CRF+Transformer实现

- 输入序列通过Embedding和Positional Encoding后，送入多层Transformer Encoder。
- 输出特征通过线性层映射为各标签的发射分数，送入CRF层进行全局解码和训练。

---

## 实验结果

1. 训练与收敛

- 训练CRF模型时，损失函数逐步收敛，表明模型参数在优化。
- 若出现“预测全为O”现象，需检查标签分布、特征模板表达能力、学习率设置等问题。

2. 预测与评估

- 使用训练好的CRF模型对验证集进行预测，输出结果到文件，并用`check`函数评估F1、Precision、Recall等指标。
- CRF+Transformer模型在特征表达和长距离依赖建模上有更好表现，适合大规模数据和复杂场景。

---

## 实验总结

1. HMM方法实现简单，适合基线对比，但对特征表达能力有限。
2. CRF通过灵活的特征模板和全局归一化，能有效提升序列标注性能。
3. CRF+Transformer结合深度上下文特征和结构化解码，进一步提升了模型表现。
4. 实验中需注意标签分布均衡、特征模板设计、学习率等超参数对模型训练和泛化能力的影响。

---

## 代码文件

- CRF.ipynb：CRF模型实现与实验主文件
- CRF_TF.ipynb：CRF+Transformer模型实现
- HMM.ipynb：HMM模型实现

---
