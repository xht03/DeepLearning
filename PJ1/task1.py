import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import pandas as pd
from lib.Net import *
from lib.Func import *
import os


architecture_1 = [
    {"module": Mlp, "params": {"input_dim": 1, "output_dim": 32, "activation": "sigmoid"}},
    {"module": Mlp, "params": {"input_dim": 32, "output_dim": 32, "activation": "sigmoid"}},
    {"module": Mlp, "params": {"input_dim": 32, "output_dim": 1, "activation": "id"}},
]

architecture_2 = [
    {"module": Flatten},
    {"module": Mlp, "params": {"input_dim": 784, "output_dim": 256, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 256, "output_dim": 64, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 64, "output_dim": 16, "activation": "relu"}},
    {"module": Mlp, "params": {"input_dim": 16, "output_dim": 10, "activation": "softmax"}},
]



datapath = "data/MNIST/train-images.idx3-ubyte"
labelpath = "data/MNIST/train-labels.idx1-ubyte"
modelpath = "model/task1.json"


def sinFit():
    # 生成训练数据
    # x: (batch_size, input_dim)
    x = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
    y = np.sin(x)

    net = Net(architecture_1)

    # 训练
    print("Start training...")

    epochs = 200
    for i in range(epochs):
        x,y = shuffle(x, y)
        net.train(x, y, batch_size=100, lr=0.005, lossfunc="square")
    
    print("Training done.")
    print("-" * 50)

    # 导出模型参数
    net.save_params(modelpath)

    # 预测
    while True:
        t = input("Input a number between -pi and pi (type 'q' to quit): ")
        if t.lower() == "q":
            break
        t = float(t)
        x_test = np.array([[t]])
        y_test = net.predict(x_test)
        y_test = y_test.item()
        print("Predicted value: %f" % y_test)
        print("True value: %f" % np.sin(t))
        print("error: %f" % abs(y_test - np.sin(t)))

    print("Prediction done.")
    print("-" * 50)

    # 计算平均误差
    y_hat = net.predict(x)
    y_hat = y_hat.flatten()
    error = np.mean(np.abs(y_hat - y))
    print("Average error: %f" % error)
    print("-" * 50)



def MNIST():
    # 读取数据
    data = idx2numpy.convert_from_file(datapath)    # (60000, 28, 28)
    data = np.expand_dims(data, axis=1)             # 添加通道维度 -> (60000, 1, 28, 28)

    # 读取标签
    label = idx2numpy.convert_from_file(labelpath)
    one_hot_labels = one_hot(label, 10)

    net = Net(architecture_2)

    # 训练
    print("Start training...")

    epochs = 50
    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        data, one_hot_labels = shuffle(data, one_hot_labels)
        net.train(data, one_hot_labels, batch_size=200, lr=0.01, lossfunc="cross_entropy")

        y_hat = net.predict(data)
        loss = cross_entropy_loss(one_hot_labels, y_hat)
        y_hat = np.argmax(y_hat, axis=1)    # (60000, 10) -> (60000,)
        accuracy = np.mean(y_hat == label)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print("-" * 50)

    print("Training done.")
    print("-" * 50)

    # 导出模型参数
    # net.save_params(modelpath)

    # 计算训练集准确率
    y_hat = net.predict(data)
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
    sinFit()