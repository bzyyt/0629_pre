import torch
from config import cfg
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, transforms


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
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_eval_set = CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    full_test_set = CIFAR10(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # 划分训练集和验证集
    split_generator = torch.Generator().manual_seed(cfg.SEED)

    train_indices = torch.randperm(len(full_train_set), generator=split_generator)
    test_indices = torch.randperm(len(full_test_set), generator=split_generator)

    train_set = Subset(full_train_set, train_indices[: cfg.TRAIN_SIZE].tolist())
    val_set = Subset(
        full_eval_set,
        train_indices[cfg.TRAIN_SIZE : cfg.TRAIN_SIZE + cfg.VAL_SIZE].tolist(),
    )
    test_set = Subset(full_test_set, test_indices[: cfg.TEST_SIZE].tolist())

    return train_set, val_set, test_set


# 转换为数据加载器
def build_dataloaders(train_set, val_set, test_set):
    train_generator = torch.Generator().manual_seed(cfg.SEED)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.use_gpu,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=cfg.use_gpu
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=cfg.use_gpu
    )
    return train_loader, val_loader, test_loader
