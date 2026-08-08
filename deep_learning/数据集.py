# %% [markdown]
# 数据集有关内容

# %%
# 依赖
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

# %%
# 下载FashionMNIST训练数据集
full_train_dataset = FashionMNIST(
    root="../data", train=True, download=True, transform=ToTensor()
)

# %%
# 下载FashionMNIST测试数据集
test_set = FashionMNIST(
    root="../data", train=False, download=True, transform=ToTensor()
)

# %%
# 划分训练集和验证集
train_set, val_set = random_split(  # 随机划分
    full_train_dataset, [54000, 6000], generator=torch.Generator().manual_seed(42)
)  # 54000个训练，6000个验证,固定随机种子保证结果一致

# DataLoader（数据加载器）是 PyTorch 中用来批量、高效地喂数据给模型的工具。
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# %%
# 一个batch的训练数据
images, labels = next(iter(train_loader))

print("训练样本数：", len(train_set))
print("验证样本数：", len(val_set))
print("测试样本数：", len(test_set))
print("图片形状：", images.shape)
print("标签形状：", labels.shape)

# %%
# 显示一个batch的训练数据
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")
plt.show()

# %%
