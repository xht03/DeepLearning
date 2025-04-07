# %%
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# %%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"device: {device}")

# %%
class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# %%
def one_hot(labels, num_classes):
    # one-hot 编码
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

# %%
datapath = "../data/MNIST/train-images.idx3-ubyte"
labelpath = "../data/MNIST/train-labels.idx1-ubyte"
modelpath = "../model/task3/MNIST.pth"

data = idx2numpy.convert_from_file(datapath)    # (60000, 28, 28)
data = np.expand_dims(data, axis=1)             # 添加通道维度 -> (60000, 1, 28, 28)
data = torch.from_numpy(data).float()

label = idx2numpy.convert_from_file(labelpath)
one_hot_labels = one_hot(label, 10)
one_hot_labels = torch.from_numpy(one_hot_labels).float()


# %%
test_datapath = "../test/MNIST/t10k-images.idx3-ubyte"
test_labelpath = "../test/MNIST/t10k-labels.idx1-ubyte"

test_data = idx2numpy.convert_from_file(test_datapath)
test_data = np.expand_dims(test_data, axis=1)
test_data = torch.from_numpy(test_data).float()

test_label = idx2numpy.convert_from_file(test_labelpath)
test_one_hot_labels = one_hot(test_label, 10)
test_one_hot_labels = torch.from_numpy(test_one_hot_labels).float()

# %%
train_dataset = TensorDataset(data, one_hot_labels)
test_dataset = TensorDataset(test_data, test_one_hot_labels)

# %%
epochs = 100            # 训练轮数

batch_size = 128        # 批大小
inital_lr = 0.001       # 初始学习率
lr_patience = 10        # 学习率衰减的耐心
lr_decay = 0.5          # 学习率衰减系数

best_accuracy = 0.0     # 最佳准确率

# %%
train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, False)

model = MNIST().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=inital_lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=lr_patience)

# %%
pbar = tqdm(range(epochs), desc="Training")
for i in pbar:
    model.train()
    running_loss = 0.0
    test_loss = 0.0
    accuracy = 0.0
    for x, y in train_loader:
        # 加载进GPU
        x = x.to(device)
        y = y.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(x)
        # 计算损失
        loss = loss_func(output, y)
        running_loss += loss.item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

    # 计算验证集损失和准确率
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            # 预测
            pred = model(x)
            # 计算损失
            loss = loss_func(pred, y)
            test_loss += loss.item()
            # 计算准确率
            accuracy += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()

    running_loss /= len(train_loader)
    test_loss /= len(test_loader)
    accuracy /= len(test_loader.dataset)
    scheduler.step(test_loss)  # 更新学习率

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), modelpath)

    pbar.set_postfix(
        loss=running_loss,
        test_loss=test_loss,
        accuracy=f"{accuracy*100:.2f}%",
        best_accuracy=f"{best_accuracy*100:.2f}%",
        lr=optimizer.param_groups[0]['lr'],
    )

# %%
print("architecture:", model)
print("param", sum(p.numel() for p in model.parameters()))
print("savepath:", modelpath)
print("best_accuracy:", best_accuracy)

# %%
# For interview
model.load_state_dict(torch.load(modelpath))

test_loss = 0.0
accuracy = 0.0

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        # 预测
        pred = model(x)
        # 计算损失
        loss = loss_func(pred, y)
        test_loss += loss.item()
        # 计算准确率
        accuracy += torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)).item()

    test_loss /= len(test_loader)
    accuracy /= len(test_loader.dataset)

print("test_loss:", test_loss)
print("test_accuracy:", f"{accuracy*100:.2f}%")


