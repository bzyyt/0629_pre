import torch
from config import cfg


# 验证或测试
def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    num_classes = cfg.NUM_CLASSES
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=cfg.device
    )

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            logits = model(X)
            loss_value = loss_fn(logits, y)

            total_loss += loss_value.item() * X.size(0)
            _, predicted = torch.max(logits, 1)
            total_samples += y.size(0)
            top1_correct += (predicted == y).sum().item()
            k = min(5, num_classes)
            top5 = torch.topk(logits, k, dim=1).indices
            top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total_correct += (predicted == y).sum().item()

            indices = y * num_classes + predicted
            confusion_matrix += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    if total_samples == 0:
        raise ValueError

    confusion_matrix = confusion_matrix.to(dtype=torch.float64)
    tp = confusion_matrix.diag()
    actual_count = confusion_matrix.sum(dim=1)
    predicted_count = confusion_matrix.sum(dim=0)

    precision = tp / predicted_count.clamp(min=1)
    recall = tp / actual_count.clamp(min=1)

    f1 = 2 * tp / (predicted_count + actual_count).clamp(min=1)

    return {
        "loss": total_loss / total_samples,
        "top1_acc": top1_correct / total_samples,
        "top5_acc": top5_correct / total_samples,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }
