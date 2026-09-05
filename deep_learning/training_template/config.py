from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    name: str = "ResNet18"

    SEED: int = 42

    # 参数
    scratch_learning_rate: float = 0.1
    feature_learning_rate: float = 0.01
    finetune_layer4_learning_rate: float = 0.0001
    finetune_fc_learning_rate: float = 0.001
    batch_size: int = 256
    SCRATCH_EPOCHS: int = 10
    FEATURE_EPOCHS: int = 5
    FINETUNE_EPOCHS: int = 5

    TRAIN_SIZE: int = 5000
    VAL_SIZE: int = 1000
    TEST_SIZE: int = 1000
    NUM_CLASSES: int = 10

    # 设备
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if use_gpu else "cpu"

    data_dir: Path = PROJECT_ROOT / "data"
    out_dir: Path = PROJECT_ROOT / "results" / "training_template"


cfg = Config()
