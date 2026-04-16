from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    data_path: str
    split_train: str = "train"
    split_val: str = "val"
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    window_size: int = 128
    stride: int = 64
    min_window_length: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    output_dir: str = "experiments/run1"
    loss_weights: dict = field(
        default_factory=lambda: {
            "rating": 1.0,
            "expected": 1.0,
            "residual": 1.0,
            "contrastive": 0.1,
        }
    )


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainConfig(**raw)


def ensure_output_dir(config):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
