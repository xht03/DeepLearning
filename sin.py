import numpy as np
import matplotlib.pyplot as plt

architecture = [
    {"input_dim": 1, "output_dim": 10, "activation": "sigmoid"},
    {"input_dim": 10, "output_dim": 1, "activation": "sigmoid"},
]

# 激活函数及其导数
def sigmod(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmod(x) * (1 - sigmod(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 损失函数
def square_loss(y_true, y_pred):
    return (1/2) * (y_true - y_pred)**2

def square_derivative(y_true, y_pred):
    return y_pred - y_true

def cross_entropy_loss(y_true, y_pred):
    m = y_pred.shape[1]     # 样本数
    loss = -1 / m * (np.dot(y_true, np.log(y_pred).T) + np.dot(1 - y_true, np.log(1 - y_pred).T))
    return np.squeeze(loss)

def cross_entropy_derivative(y_true, y_pred):
    return - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))

# 初始化参数
def init_params(architecture):
    params = {}

    for idx, layer in enumerate(architecture):
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]

        params["w"+str(idx + 1)] = np.random.randn(output_dim, input_dim) * 0.1
        params["b"+str(idx + 1)] = np.random.randn(output_dim, 1) * 0.1
    
    return params

# 单层前向传播
def single_forward(x, w, b, activation="sigmoid"):
    if activation == "sigmoid":
        activation_func = sigmod
    elif activation == "relu":
        activation_func = relu
    else:
        raise Exception("Non-supported activation function")

    z = np.dot(w, x) + b
    o = activation_func(z)
    return z, o

# 多层前向传播
def forward(x, params, architecture):
    memory = {}
    i = x;

    for idx, layer in enumerate(architecture):
        activation = layer["activation"]
        w = params["w" + str(idx + 1)]
        b = params["b" + str(idx + 1)]
        z, o = single_forward(i, w, b, activation)

        memory["i"+str(idx + 1)] = i;
        memory["z"+str(idx + 1)] = z;
        memory["o"+str(idx + 1)] = o;
        i = o;

    return i, memory

# 单层反向传播
# d: 来自后一层的梯度
# w: 来自当前层的权重
# b: 来自当前层的偏置
# o: 当前层的激活值
# z: 当前层的输出
# i: 当前层的输入
# activation: 当前层的激活函数
def single_backward(d,w,b,o,z,i,activation="sigmoid"):
    m = i.shape[1]      # 样本数

    if activation == "sigmoid":
        derivative_func = sigmoid_derivative
    elif activation == "relu":
        derivative_func = relu_derivative
    else:
        raise Exception("Non-supported activation function")

    dz = d * derivative_func(z)
    dw = 1 / m * np.dot(dz, i.T)
    db = 1 / m * np.sum(dz, axis=1, keepdims=True)
    di = np.dot(w.T, dz)
    
    return di, dw, db

# 多层反向传播
def backward(y_hat, y, memory, params, architecture):
    grades = {}
    m = y.shape[1]    # 样本数
    y = y.reshape(y_hat.shape)

    dl = cross_entropy_derivative(y, y_hat)     # 损失函数对输出层的导数（梯度）

    dprev = dl

    for idx, layer in reversed(list(enumerate(architecture))):
        activation = layer["activation"]
        i = memory["i"+str(idx + 1)]
        z = memory["z"+str(idx + 1)]
        o = memory["o"+str(idx + 1)]
        w = params["w"+str(idx + 1)]
        b = params["b"+str(idx + 1)]

        dprev, dw, db = single_backward(dprev, w, b, o, z, i, activation)

        grades["dw"+str(idx + 1)] = dw
        grades["db"+str(idx + 1)] = db

    return grades

# 更新参数
def update(params, grades, architecture, lr):
    for idx, layer in enumerate(architecture):
        params["w"+str(idx + 1)] -= lr * grades["dw"+str(idx + 1)]
        params["b"+str(idx + 1)] -= lr * grades["db"+str(idx + 1)]

    return params

# 训练
def train(x, y, architecture, epochs, lr):
    params = init_params(architecture)
    loss_history = []

    for epoch in range(epochs):
        y_hat, memory = forward(x, params, architecture)
        loss = cross_entropy_loss(y, y_hat)
        loss_history.append(loss)
        grades = backward(y_hat, y, memory, params, architecture)
        params = update(params, grades, architecture, lr)

        if epoch % 100 == 0:
            print("Epoch %d, loss: %f" % (epoch, loss))

    return params, loss_history

def main():
    # 生成训练数据
    x = np.linspace(-np.pi, np.pi, 1000).reshape(1, -1)     # (1, 1000) 列向量
    y = np.sin(x)
    print("Training data prepared. Data shape: (%d, %d)" % x.shape, y.shape)

    # 训练
    print("Start training...")
    epochs = 10000
    lr = 0.01
    params, loss_history = train(x, y, architecture, epochs, lr)
    print("Training done.")

    # 预测
    t = input("Input a number between -pi and pi: ")
    t = float(t)
    x_test = np.array([[t]])
    y_test, _ = forward(x_test, params, architecture)
    print("Predicted value: %f" % y_test)
    print("True value: %f" % np.sin(t))

    # 绘制损失曲线
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

if __name__ == "__main__":
    main()



