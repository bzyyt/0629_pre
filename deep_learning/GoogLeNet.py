import copy

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm.auto import tqdm

netname = "GoogLeNet"
# 超参数
batch_size = 256
num_epochs = 30
config = [
    {
        "name": f"{netname}",
        "learning_rate": 0.1,
    },
]
path = "results/week06_transfer_learning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(f"Using device: {device}, GPU available: {use_gpu}")

# 转换数据尺寸
transform = Compose([Resize((96, 96)), ToTensor()])

# 下载训练数据集
full_train_set = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
# 下载测试数据集
test_set = FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# 划分训练集和验证集
train_set, val_set = random_split(
    full_train_set, [54000, 6000], generator=torch.Generator().manual_seed(42)
)

# 转换为数据加载器
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False, pin_memory=use_gpu
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=use_gpu
)


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        # 1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 3x3最大池化层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = torch.relu(self.p1_1(x))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(x))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(x))))
        p4 = torch.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


# 初始化模型
def model_factory():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    net.apply(init_weights)
    return net


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# 训练函数
def train_model(config):
    result = {}
    result["config"] = config["name"]
    result["learning_rate"] = config["learning_rate"]
    torch.manual_seed(42)
    # 创建模型
    net = model_factory().to(device)  # 模型移动到GPU
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction="mean")
    # 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=config["learning_rate"])
    # 重新创建训练数据加载器以确保每次训练时数据顺序相同
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        pin_memory=use_gpu,
    )

    # 最好的模型
    best_model = copy.deepcopy(net.state_dict())
    best_val_acc = float("-inf")
    best_epoch = 0
    best_train_acc = 0.0

    # loss历史
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0
        # 每个批次的进度条
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
        )
        for X, y in train_bar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # 数据移动到GPU
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
        train_accuracies.append(train_acc)
        net.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = net(X)
                loss_value = loss(logits, y)
                total_loss += loss_value.item() * y.size(0)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        val_loss = total_loss / total
        val_losses.append(val_loss)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_train_acc = train_acc
            best_model = copy.deepcopy(net.state_dict())

        val_accuracies.append(val_acc)
        print(
            f"Config: {config['name']}, "
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
        )

    result["best_epoch"] = best_epoch
    result["best_train_acc"] = best_train_acc
    result["best_val_acc"] = best_val_acc

    # 测试
    net.load_state_dict(best_model)
    net.eval()
    state_dict_cpu = {
        name: value.detach().cpu().clone() for name, value in net.state_dict().items()
    }
    torch.save(
        state_dict_cpu, f"{path}/{config['name']}_best_model.pth"
    )  # 保存最佳模型
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = net(X)
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    test_acc = correct / total
    print(f"Test Acc: {test_acc:.4f}")

    result["test_acc"] = test_acc

    # 绘制损失曲线
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, marker="s", label="Val Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}/{config['name']}_loss.jpg")
    plt.show()

    # 绘制准确率曲线
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker="o", label="Train Acc")
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker="s", label="Val Acc")
    plt.title("Training and Validation Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{path}/{config['name']}_accuracy.jpg")
    plt.show()

    return result


all_results = []
for cfg in config:
    result = train_model(cfg)
    all_results.append(result)

# 准确率表格
result_table = pd.DataFrame(all_results)
result_table.to_csv(f"{path}/{netname}_results.csv", index=False, encoding="utf-8-sig")
