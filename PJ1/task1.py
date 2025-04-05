import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import pandas as pd
from lib.Net import *
from lib.Func import *
import os

architecture = [
    {"module": Flatten},
    {"module": Mlp, "params": {"input_dim": 784, "output_dim": 256, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 256, "output_dim": 64, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 64, "output_dim": 16, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 16, "output_dim": 10, "activation": "softmax"}},
]

datapath = "data/MNIST/train-images.idx3-ubyte"
labelpath = "data/MNIST/train-labels.idx1-ubyte"
modelpath = "model/task1.json"

def MNIST():
    # 读取数据
    data = idx2numpy.convert_from_file(datapath)    # (60000, 28, 28)
    data = np.expand_dims(data, axis=1)             # 添加通道维度 -> (60000, 1, 28, 28)
    data = data.astype(np.float64)                  # 转换为 float64

    # 读取标签
    label = idx2numpy.convert_from_file(labelpath)
    one_hot_labels = one_hot(label, 10)

    net = Net(architecture)

    # 训练
    print("Start training...")
    net.train(data, one_hot_labels, epochs=50, batch_size=100, lr=0.01, lossfunc="cross_entropy")
    print("Training done.")
    print("-" * 50)

    # 导出模型参数
    net.save_params(modelpath)

    # 计算训练集准确率
    y_hat = net.forward(data)
    y_hat = np.argmax(y_hat, axis=1)    # (60000, 10) -> (60000,)
    accuracy = np.mean(y_hat == label)
    print("Train accuracy: ", accuracy)

    # 随机抽样10个预测结果
    sample_indices = np.random.choice(data.shape[0], 10, replace=False)
    sample_images = data[sample_indices]
    sample_labels = label[sample_indices]
    sample_predictions = y_hat[sample_indices]

    # 绘制预样例图像
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i][0], cmap='gray')
        plt.title(f"True: {sample_labels[i]}, Pred: {sample_predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    MNIST()