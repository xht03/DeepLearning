import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import pandas as pd
from lib.Func import * 
from lib.MLP import *
import os

architecture_2 = [
    {"input_dim": 784, "output_dim": 256, "activation": "relu"},
    {"input_dim": 256, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 10, "activation": "softmax"},
]


# 读取并预处理图形数据数据
def preprocessData(datapath, labelpath):
    # 读取数据
    data = idx2numpy.convert_from_file(datapath)
    labels = idx2numpy.convert_from_file(labelpath)

    print("Data shape:" + str(data.shape))
    print("Labels shape:" + str(labels.shape))

    # 转换为列向量
    data = data.reshape(data.shape[0], -1).T
    labels = labels.reshape(1, labels.shape[0])
    # one-hot 编码
    on_hot_labels = np.zeros((10, labels.shape[1]))
    on_hot_labels[labels[0], np.arange(labels.shape[1])] = 1

    return data, labels, on_hot_labels


# 手写数字识别
def digitRecognition():
    # 读取数据
    datapath = "data/MNIST/train-images.idx3-ubyte"
    labelpath = "data/MNIST/train-labels.idx1-ubyte"
    data, labels, on_hot_labels = preprocessData(datapath, labelpath)

    choice = input("Training or Test ? (1 for train, 2 for test): ")
    if choice == "1":
        # 训练
        print("Start training...")
        params, loss_history = train(data, on_hot_labels, architecture_2, epochs=15, batch_size=100, lr=0.01, lossfunc="cross_entropy")
        print("Training done.")
        print("-" * 50)

        # 导出模型参数
        output_dir = "model"
        model_name = "digits_model.npz"
        exportParams(params, output_dir, model_name)


        # 计算训练集准确率
        y_hat, _ = forward(data, params, architecture_2)
        y_hat = np.argmax(y_hat, axis=0)
        accuracy = np.mean(y_hat == labels)
        print("Total training samples: %d" % data.shape[1])
        print("Training accuracy: %f" % accuracy)
        print("-" * 50)

        # 随机抽样10个预测结果
        while True:
            results = []
            indices = np.random.choice(data.shape[1], 10, replace=False)
            for i in indices:
                x_test = data[:, i].reshape(-1, 1)
                y_test, _ = forward(x_test, params, architecture_2)
                y_test = np.argmax(y_test)
                results.append((i, y_test, labels[0, i]))

            # 绘制样例图像
            fig, axes = plt.subplots(2, 5, figsize=(10, 5))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(data[:, results[i][0]].reshape(28, 28), cmap="gray")
                ax.set_title(f"Pred: {results[i][1]}, True: {results[i][2]}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()

            # 是否继续
            t = input("Continue? (y/n): ")
            if t.lower() != "y":
                break
        
        # 绘制损失曲线
        loss_history = np.array(loss_history)
        loss_history = np.mean(loss_history, axis=(1,2))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

    elif choice == "2":
        modelpath = "model/digits_model.npz"
        params = importParams(modelpath)

        # 计算训练集准确率
        y_hat, _ = forward(data, params, architecture_2)
        y_hat = np.argmax(y_hat, axis=0)
        accuracy = np.mean(y_hat == labels)
        print("Total training samples: %d" % data.shape[1])
        print("Training accuracy: %f" % accuracy)
        print("-" * 50)



if __name__ == "__main__":
    digitRecognition()