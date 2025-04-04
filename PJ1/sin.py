import numpy as np
import matplotlib.pyplot as plt
from lib.Func import * 
from lib.MLP import *

architecture_1 = [
    {"input_dim": 1, "output_dim": 32, "activation": "sigmoid"},
    {"input_dim": 32, "output_dim": 32, "activation": "sigmoid"},
    {"input_dim": 32, "output_dim": 1, "activation": "id"},
]

# 拟合 sin 函数
def sinFit():
    # 生成训练数据
    x = np.linspace(-np.pi, np.pi, 1000).reshape(1, -1)  # (1, 1000) 列向量
    y = np.sin(x)
    print("Training data prepared. Input data shape: (%d, %d)" % x.shape)
    print("-" * 50)

    # 训练
    print("Start training...")
    params, loss_history = train(x, y, architecture_1, epochs=200000, batch_size=100, lr=0.005, lossfunc="square")
    print("Training done.")
    print("-" * 50)

    # 导出模型参数
    output_dir = "model"
    model_name = "sin_model.npz"
    exportParams(params, output_dir, model_name)
    print("Model parameters exported to %s/%s" % (output_dir, model_name))
    print("-" * 50)

    # 预测
    while True:
        t = input("Input a number between -pi and pi (type 'q' to quit): ")
        if t.lower() == "q":
            break
        t = float(t)
        x_test = np.array([[t]])
        y_test, _ = forward(x_test, params, architecture_1)
        y_test = y_test.item()
        print("Predicted value: %f" % y_test)
        print("True value: %f" % np.sin(t))
        print("error: %f" % abs(y_test - np.sin(t)))

    print("Prediction done.")
    print("-" * 50)

    # 计算平均误差
    y_hat, _ = forward(x, params, architecture_1)
    y_hat = y_hat.flatten()
    error = np.mean(np.abs(y_hat - y))
    print("Average error: %f" % error)
    print("-" * 50)

    # 绘制拟合曲线
    x_fit = np.linspace(-np.pi, np.pi, 1000).reshape(1, -1)
    y_fit, _ = forward(x_fit, params, architecture_1)
    y_fit = y_fit.flatten()
    plt.figure()
    plt.plot(x.flatten(), y.flatten(), label="True Function")
    plt.plot(x_fit.flatten(), y_fit, label="Fitted Function")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Fitting")
    plt.show()

    # 绘制损失曲线
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()


if __name__ == "__main__":
    sinFit()
