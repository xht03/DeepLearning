import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import idx2numpy

architecture = [
    {"input_dim": 1, "output_dim": 8, "activation": "sigmoid"},
    {"input_dim": 8, "output_dim": 8, "activation": "sigmoid"},
    {"input_dim": 8, "output_dim": 1, "activation": "id"},
]

def shuffle(x, y):
    indices = np.random.permutation(x.shape[1])
    return x[:, indices], y[:, indices]

# 激活函数及其导数
def id(x):
    return x

def id_derivative(x):
    return 1

def sigmod(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmod(x) * (1 - sigmod(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 损失函数
def square_loss(y_true, y_pred):
    return np.mean((1/2) * (y_true - y_pred)**2)

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
    if activation == "id":
        activation_func = id
    elif activation == "sigmoid":
        activation_func = sigmod
    elif activation == "relu":
        activation_func = relu
    else:
        raise Exception("Non-supported activation function")

    z = np.dot(w, x) + b
    o = activation_func(z)
    return z, o

# 多层前向传播
def forward(x, params, architecture, softmax=False):
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
        
        i = o
    
    if softmax:
        i = softmax(i)

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

    if activation == "id":
        derivative_func = id_derivative
    elif activation == "sigmoid":
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
def backward(y_hat, y, memory, params, architecture, softmax=False):
    grades = {}
    m = y.shape[1]    # 样本数
    y = y.reshape(y_hat.shape)

    if softmax:
        dl = cross_entropy_derivative(y, y_hat)     # 损失函数对输出层的导数（梯度）
    else:
        dl = square_derivative(y, y_hat)

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
def train(x, y, architecture, epochs, lr, softmax=False):
    params = init_params(architecture)
    loss_history = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        y_hat, memory = forward(x, params, architecture, softmax)
        if softmax:
            loss = cross_entropy_loss(y, y_hat)
        else:
            loss = square_loss(y, y_hat)
        loss_history.append(loss)
        grades = backward(y_hat, y, memory, params, architecture, softmax)
        params = update(params, grades, architecture, lr)

    return params, loss_history

# 拟合 sin 函数
def sinFit():
    # 生成训练数据
    x = np.linspace(-np.pi, np.pi, 1000).reshape(1, -1)     # (1, 1000) 列向量
    y = np.sin(x)
    print("Training data prepared. Input data shape: (%d, %d)" % x.shape)

    # 训练
    print("Start training...")
    epochs = 100000
    lr = 0.01
    params, loss_history = train(x, y, architecture, epochs, lr)
    print("Training done.")

    # 预测
    while True:
        t = input("Input a number between -pi and pi (type 'bye' to quit): ")
        if t.lower() == "bye":
            break
        t = float(t)
        x_test = np.array([[t]])
        y_test, _ = forward(x_test, params, architecture)
        y_test = y_test.item()
        print("Predicted value: %f" % y_test)
        print("True value: %f" % np.sin(t))
        print("error: %f %%" % abs(y_test - np.sin(t)))
        print("-" * 50)

    print("Prediction done.")

    # 打印模型参数
    for idx, layer in enumerate(architecture):
        print("Layer %d:" % (idx + 1))
        print("Weights: ", params["w" + str(idx + 1)])
        print("Biases: ", params["b" + str(idx + 1)])

    # 绘制拟合曲线
    x_fit = np.linspace(-np.pi, np.pi, 1000).reshape(1, -1)
    y_fit, _ = forward(x_fit, params, architecture)
    y_fit = y_fit.flatten()

    plt.figure()
    plt.plot(x.flatten(), y.flatten(), label='True Function')
    plt.plot(x_fit.flatten(), y_fit, label='Fitted Function')
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


# 手写数字识别
def digitRecognition():
    # 读取训练数据
    data = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
    labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
    # 转换为列向量
    data = data.reshape(data.shape[0], -1).T
    labels = labels.reshape(1, labels.shape[0])
    # one-hot 编码
    on_hot_labels = np.zeros((10, labels.shape[1]))
    on_hot_labels[labels[0], np.arange(labels.shape[1])] = 1

    print("Training data prepared. Input data shape: (%d, %d)" % data.shape)

    # 训练
    print("Start training...")
    epochs = 100
    lr = 0.01
    params, loss_history = train(data, labels, architecture, epochs, lr)
    print("Training done.")

    # 打印模型参数
    for idx, layer in enumerate(architecture):
        print("Layer %d:" % (idx + 1))
        print("Weights: ", params["w" + str(idx + 1)])
        print("Biases: ", params["b" + str(idx + 1)])
    
    # 计算训练集准确率
    y_hat, _ = forward(data, params, architecture, softmax=True)
    y_hat = np.argmax(y_hat, axis=0)
    accuracy = np.mean(y_hat == labels)
    print("Training accuracy: %f" % accuracy)

    


if __name__ == "__main__":
    digitRecognition()



