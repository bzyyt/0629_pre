from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    name: str = "flowers102"

    SEED: int = 42

    # 参数
    scratch_learning_rate: float = 0.1
    feature_learning_rate: float = 0.01
    finetune_layer4_learning_rate: float = 0.0001
    finetune_fc_learning_rate: float = 0.001
    batch_size: int = 32
    SCRATCH_EPOCHS: int = 30
    FEATURE_EPOCHS: int = 30
    FINETUNE_EPOCHS: int = 30

    # 对照实验：在原有随机裁剪、水平翻转之外增加轻度颜色扰动。
    jitter_brightness: float = 0.15
    jitter_contrast: float = 0.15
    jitter_saturation: float = 0.10

    NUM_CLASSES: int = 102

    # 设备
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if use_gpu else "cpu"

    data_dir: Path = PROJECT_ROOT / "data"
    out_dir: Path = PROJECT_ROOT / "results" / "flowers102"


cfg = Config()
