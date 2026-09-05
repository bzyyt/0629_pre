import matplotlib.pyplot as plt
from config import cfg


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
    plt.savefig(f"{cfg.out_dir}/{cfg.name}_{name}_loss.png")
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
    plt.savefig(f"{cfg.out_dir}/{cfg.name}_{name}_accuracy.png")
    plt.close()
