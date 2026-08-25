import copy
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Resize, ToTensor

netname = "VGG"
# 超参数
batch_size = 256
num_epochs = 20
config = [
    {
        "name": "VGG",
        "learning_rate": 0.05,
    },
]
path = "results/week06_transfer_learning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(f"Using device: {device}, GPU available: {use_gpu}")

# 转换数据尺寸
transform = Compose([Resize((224, 224)), ToTensor()])

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


# 时间格式化
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def vgg_block(num_convs, in_channels, out_channels):
    layers = []

    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 初始化模型
def model_factory(conv_arch):
    conv_blks = []
    in_channels = 1

    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),
    )
    net.apply(init_weights)
    return net


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 训练函数
def train_model(config):
    result = {}
    result["config"] = config["name"]
    result["learning_rate"] = config["learning_rate"]
    torch.manual_seed(42)
    # 创建模型
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = model_factory(small_conv_arch).to(device)  # 模型移动到GPU
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

    # 记录开始时间
    training_start_time = time.perf_counter()

    # 训练
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for X, y in train_loader:
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

        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - training_start_time
        average_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - epoch - 1
        estimated_remaining_time = average_epoch_time * remaining_epochs

        val_accuracies.append(val_acc)
        print(
            f"Config: {config['name']}, "
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Epoch Time: {format_time(epoch_time)}, "
            f"Elapsed: {format_time(elapsed_time)}, "
            f"ETA: {format_time(estimated_remaining_time)}"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total Training Time: {format_time(total_training_time)}")

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
