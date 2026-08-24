import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

path = "results/week05_cnn_baseline"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )


def collect_examples(net, data_loader, num_samples=10):
    correct_examples = []
    error_examples = []

    net.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            logits = net(images.to(device))
            predictions = logits.argmax(dim=1).cpu()

            for image, true_label, predicted_label in zip(images, labels, predictions):
                example = (
                    image.squeeze(0),
                    true_label.item(),
                    predicted_label.item(),
                )

                if true_label == predicted_label:
                    if len(correct_examples) < num_samples:
                        correct_examples.append(example)
                else:
                    if len(error_examples) < num_samples:
                        error_examples.append(example)

            if (
                len(correct_examples) >= num_samples
                and len(error_examples) >= num_samples
            ):
                break

    return correct_examples, error_examples


def save_examples(examples, save_path, title, title_color):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = list(axes.flat)

    for ax, (image, true_label, predicted_label) in zip(axes, examples):
        ax.imshow(image.numpy(), cmap="gray")
        ax.set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}",
            color=title_color,
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(examples) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


test_set = FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_loader = DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
)

# 创建相同的模型结构
net = build_model()

# 加载权重
state_dict = torch.load(
    f"{path}/cnn_learning_rate_0.9_best_model.pth",
    map_location="cpu",
    weights_only=True,
)

net.load_state_dict(state_dict)
net = net.to(device)
net.eval()

correct_examples, error_examples = collect_examples(
    net,
    test_loader,
    num_samples=10,
)

save_examples(
    correct_examples,
    f"{path}/prediction_samples.jpg",
    "Correct Prediction Samples",
    "green",
)

save_examples(
    error_examples,
    f"{path}/error_samples.jpg",
    "Incorrect Prediction Samples",
    "red",
)
