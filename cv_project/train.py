import copy

import torch
from config import cfg
from evaluate import evaluate_model
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from tqdm import tqdm


# 计算需要训练的参数数量
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    model.to(cfg.device)

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

    if cfg.use_gpu:
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
            X = X.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)  # 数据移动到GPU

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

    if cfg.use_gpu:
        torch.cuda.synchronize()

    model.load_state_dict(best_model)
    model.eval()
    torch.save(model.state_dict(), f"{cfg.out_dir}/{name}_best_model.pth")

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
