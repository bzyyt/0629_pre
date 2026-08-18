import os

import certifi
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor

# HTTPS证书
os.environ["SSL_CERT_FILE"] = certifi.where()

# 超参数
batch_size = 256
num_epochs = 10
learning_rate = 0.1

# 下载训练数据集
full_train_set = KMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# 下载测试数据集
test_set = KMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 划分训练集和验证集
train_set, val_set = random_split(
    full_train_set, [54000, 6000], generator=torch.Generator().manual_seed(42)
)

# 转换为数据加载器
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 初始化模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 损失函数
loss = nn.CrossEntropyLoss(reduction="mean")

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# loss历史
train_losses = []

# 训练
for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in train_loader:
        logits = net(X)
        loss_value = loss(logits, y)
        trainer.zero_grad()
        loss_value.backward()
        trainer.step()
        total_loss += loss_value.item() * X.size(0)
        _, predicted = torch.max(logits, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    train_loss = total_loss / total
    train_losses.append(train_loss)
    train_acc = correct / total

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            logits = net(X)
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    val_acc = correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, "
        f"Val Acc: {val_acc:.4f}"
    )

# 测试
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_loader:
        logits = net(X)
        _, predicted = torch.max(logits, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
test_acc = correct / total
print(f"Test Acc: {test_acc:.4f}")


# 绘制训练损失曲线
plt.plot(range(1, num_epochs + 1), train_losses, marker="o")
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/week03_dl_basics/training_loss.jpg")
plt.show()
