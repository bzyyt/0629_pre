import copy

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet
from torchvision.transforms import Compose, Resize
from tqdm.auto import tqdm

netname = "ResNet_pretrained"
# 网络参数
SEED = 42
TRAIN_SIZE = 5000
VAL_SIZE = 1000
TEST_SIZE = 1000
NUM_CLASSES = 10

# 超参数
batch_size = 256
SCRATCH_EPOCHS = 10
FEATURE_EPOCHS = 5
FINETUNE_EPOCHS = 5
path = "results/week06_transfer_learning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(f"Using device: {device}, GPU available: {use_gpu}")


# 加载数据集
def build_datasets():
    # 标准化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # 数据增强
    train_transform = Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = Compose([
        Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # 加载数据集
    full_train_set = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    full_eval_set = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=eval_transform,
    )
    full_test_set = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform,
    )

    # 划分训练集和验证集
    split_generator = torch.Generator().manual_seed(SEED)

    train_indices = torch.randperm(len(full_train_set), generator=split_generator)
    test_indices = torch.randperm(len(full_test_set), generator=split_generator)

    train_set = Subset(full_train_set, train_indices[:TRAIN_SIZE].tolist())
    val_set = Subset(
        full_eval_set, train_indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE].tolist()
    )
    test_set = Subset(full_test_set, test_indices[:TEST_SIZE].tolist())

    return train_set, val_set, test_set


# 转换为数据加载器
def build_dataloaders(train_set, val_set, test_set):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, pin_memory=use_gpu
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=use_gpu
    )
    return train_loader, val_loader, test_loader


# 模型构建
# 空模型
def build_scratch_model() -> ResNet:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# 预训练模型，冻结卷积层，只训练全连接层
def build_feature_extractor_model() -> ResNet:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# 解冻最后一层卷积层，训练全连接层和最后一层卷积层
def unfreeze_layer4(model: ResNet) -> ResNet:
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model


# 不同训练模式
def set_train_mode(model: ResNet, mode: str):
    if mode == "all":
        model.train()
    elif mode == "fc":
        model.eval()
        model.fc.train()
    elif mode == "layer4":
        model.eval()
        model.layer4.train()
        model.fc.train()
    else:
        raise ValueError(f"Unknown mode: {mode}")


# 计算需要训练的参数数量
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 验证或测试
def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(X)
            loss_value = loss_fn(logits, y)
            total_loss += loss_value.item() * X.size(0)
            _, predicted = torch.max(logits, 1)
            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()
    return total_loss / total_samples, total_correct / total_samples


# 单个训练循环
def train_model(
    name: str,
    model: ResNet,
    train_scope: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
):
    model.to(device)

    trainable_parameters = count_trainable_parameters(model)

    # 优化器
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    print()
    print("=" * 60)
    print(f"实验：{name}")
    print(f"可训练参数量：{trainable_parameters:,}")
    print("=" * 60)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_epoch = 0
    best_train_acc = 0.0
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())

    if use_gpu:
        torch.cuda.synchronize()

    for epoch in range(num_epochs):
        set_train_mode(model, mode=train_scope)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
        )

        for X, y in train_bar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # 数据移动到GPU

            logits = model(X)
            loss_value = loss(logits, y)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item() * X.size(0)
            _, predicted = torch.max(logits, 1)
            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()

            train_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc = evaluate_model(model, val_loader, loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_epoch = epoch + 1
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{name} |"
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step()

    if use_gpu:
        torch.cuda.synchronize()

    model.load_state_dict(best_model)
    model.eval()
    torch.save(model.state_dict(), f"{path}/{name}_best_model.pth")

    test_loss, test_acc = evaluate_model(model, test_loader, loss)

    print(
        f"{name} | Best Epoch: {best_epoch}, "
        f"Best Train Acc: {best_train_acc:.4f}, "
        f"Best Val Acc: {best_val_acc:.4f}, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )

    result = {
        "name": name,
        "train_scope": train_scope,
        "best_epoch": best_epoch,
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    return model, history, result


# 绘制曲线
def plot_history(history, name):
    epochs = range(1, len(history["train_loss"]) + 1)

    # loss
    plt.figure(figsize=(6, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train Loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        marker="s",
        label="Val Loss",
    )
    plt.title(f"{name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path}/{netname}_{name}_loss.png")
    plt.close()

    # accuracy
    plt.figure(figsize=(6, 5))
    plt.plot(
        epochs,
        history["train_acc"],
        marker="o",
        label="Train Acc",
    )
    plt.plot(
        epochs,
        history["val_acc"],
        marker="s",
        label="Val Acc",
    )
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path}/{netname}_{name}_accuracy.png")
    plt.close()


# 保存结果到 CSV
def save_results_to_csv(results):
    df = pd.DataFrame(results)
    df.to_csv(f"{path}/{netname}_results.csv", index=False)


# main
train_set, val_set, test_set = build_datasets()

all_results = []

# 实验一：从头训练 ResNet18
train_loader, val_loader, test_loader = build_dataloaders(train_set, val_set, test_set)
scratch_model = build_scratch_model()
scratch_optimizer = torch.optim.SGD(
    scratch_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
)

scratch_model, scratch_history, scratch_result = train_model(
    name="scratch",
    model=scratch_model,
    train_scope="all",
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=scratch_optimizer,
    num_epochs=SCRATCH_EPOCHS,
)

plot_history(scratch_history, "scratch")
all_results.append(scratch_result)

del scratch_model
if use_gpu:
    torch.cuda.empty_cache()

# 实验二：使用预训练模型，冻结卷积层，只训练全连接层
train_loader, val_loader, test_loader = build_dataloaders(train_set, val_set, test_set)
feature_extractor_model = build_feature_extractor_model()
feature_extractor_optimizer = torch.optim.SGD(
    feature_extractor_model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
)

feature_extractor_model, feature_extractor_history, feature_extractor_result = (
    train_model(
        name="feature_extractor",
        model=feature_extractor_model,
        train_scope="fc",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=feature_extractor_optimizer,
        num_epochs=FEATURE_EPOCHS,
    )
)

plot_history(feature_extractor_history, "feature_extractor")
all_results.append(feature_extractor_result)

# 实验三：解冻最后一层卷积层，训练全连接层和最后一层卷积层
unfreeze_layer4(feature_extractor_model)
fine_tune_optimizer = torch.optim.SGD(
    [
        {"params": feature_extractor_model.layer4.parameters(), "lr": 0.0001},
        {"params": feature_extractor_model.fc.parameters(), "lr": 0.001},
    ],
    momentum=0.9,
    weight_decay=1e-4,
)

transfer_model, transfer_history, transfer_result = train_model(
    name="fine_tune",
    model=feature_extractor_model,
    train_scope="layer4",
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=fine_tune_optimizer,
    num_epochs=FINETUNE_EPOCHS,
)

plot_history(transfer_history, "fine_tune")
all_results.append(transfer_result)

# 保存所有结果到 CSV
save_results_to_csv(all_results)
