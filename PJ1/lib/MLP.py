import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.Func import *
import os

# 打乱数据
def shuffle(x, y):
    indices = np.random.permutation(x.shape[1])
    return x[:, indices], y[:, indices]

# 初始化参数
def init_params(architecture):
    params = {}

    for idx, layer in enumerate(architecture):
        input_dim = layer["input_dim"]
        output_dim = layer["output_dim"]

        params["w" + str(idx + 1)] = np.random.randn(output_dim, input_dim) * 0.01
        params["b" + str(idx + 1)] = np.random.randn(output_dim, 1) * 0.01

    return params

# 单层前向传播
def single_forward(x, w, b, activation="sigmoid"):
    if activation == "id":
        activation_func = id
    elif activation == "sigmoid":
        activation_func = sigmoid
    elif activation == "relu":
        activation_func = relu
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception("Non-supported activation function")

    z = np.dot(w, x) + b
    o = activation_func(z)
    return z, o

# 多层前向传播
def forward(x, params, architecture):
    memory = {}
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


    dz = d * derivative_func(z)
    dw = 1 / m * np.dot(dz, i.T)
    db = 1 / m * np.sum(dz, axis=1, keepdims=True)
    di = np.dot(w.T, dz)

    return di, dw, db

# 多层反向传播
def backward(y_hat, y, memory, params, architecture, lossfunc="squre"):
    grades = {}
    m = y.shape[1]  # 样本数
    y = y.reshape(y_hat.shape)

    if lossfunc == "cross_entropy":
        dl = cross_entropy_derivative(y, y_hat)  # 损失函数对输出层的导数（梯度）
    elif lossfunc == "square":
        dl = square_derivative(y, y_hat)
    else:
        raise Exception("Non-supported loss function")

    dprev = dl

    for idx, layer in reversed(list(enumerate(architecture))):
        activation = layer["activation"]
        i = memory["i" + str(idx + 1)]
        z = memory["z" + str(idx + 1)]
        o = memory["o" + str(idx + 1)]
        w = params["w" + str(idx + 1)]
        b = params["b" + str(idx + 1)]

        dprev, dw, db = single_backward(dprev, w, b, o, z, i, activation)

        grades["dw" + str(idx + 1)] = dw
        grades["db" + str(idx + 1)] = db

    return grades

# 更新参数
def update(params, grades, architecture, lr):
    for idx, layer in enumerate(architecture):
        params["w" + str(idx + 1)] -= lr * grades["dw" + str(idx + 1)]
        params["b" + str(idx + 1)] -= lr * grades["db" + str(idx + 1)]

    return params

# 训练
def train(x, y, architecture, epochs, batch_size, lr, lossfunc="square"):
    params = init_params(architecture)  # 初始化参数
    loss_history = []  # 记录每个 epoch 的损失值
    m = x.shape[1]  # 样本数

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        x, y = shuffle(x, y)  # 每个 epoch 随机打乱数据
        epoch_loss = 0  # 记录每个 epoch 的损失值

        for i in range(0, m, batch_size):
            x_batch = x[:, i : i + batch_size]
            y_batch = y[:, i : i + batch_size]

            y_hat, memory = forward(x_batch, params, architecture)
            
            if lossfunc == "cross_entropy":
                loss = cross_entropy_loss(y_batch, y_hat)
            elif lossfunc == "square":
                loss = square_loss(y_batch, y_hat)
            else:
                raise Exception("Non-supported loss function")

            epoch_loss += loss
            grades = backward(y_hat, y_batch, memory, params, architecture, lossfunc)
            params = update(params, grades, architecture, lr)

        loss_history.append(epoch_loss / (m//batch_size))

    return params, loss_history


# 导出模型参数
def exportParams(params, outputdir, modelname):
    os.makedirs(outputdir, exist_ok=True)
    np.savez(os.path.join(outputdir, modelname), **params)
    print(f"Model parameters saved to {os.path.join(outputdir, modelname)}")


# 导入模型参数
def importParams(modelpath):
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"Model file {modelpath} does not exist.")
    params = np.load(modelpath)
    return {key: value for key, value in params.items()}