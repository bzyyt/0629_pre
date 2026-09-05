import torch
from config import cfg


# 验证或测试
def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            logits = model(X)
            loss_value = loss_fn(logits, y)
            total_loss += loss_value.item() * X.size(0)
            _, predicted = torch.max(logits, 1)
            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()
    return total_loss / total_samples, total_correct / total_samples
