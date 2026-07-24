from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    img_size: list[int] = field(default_factory=lambda: [384, 384])
    crop_bbox: bool = True
    bbox_pad: float = 0.05  # <-- Добавлено
    num_workers: int = 4
    target_noise: float = 0.05

    use_soft_labels: bool = False  # <-- Добавлено
    soft_label_sigma: float = 0.25  # <-- Добавлено
    use_balanced_sampler: bool = True  # <-- Добавлено
    class_weight_beta: float = 0.999  # <-- Добавлено


@dataclass
class ModelConfig:
    name: str = "vit_small_patch16_dinov3.lvd1689m"
    pretrained: bool = True
    freeze_backbone: bool = False  # <-- Добавлено
    use_cls_token: bool = True  # <-- Добавлено
    use_patch_tokens: bool = False  # <-- Добавлено
    patch_pool: str = "avg"  # <-- Добавлено
    drop_rate: float = 0.3
    # Разрешаем либо float (например, 2.88), либо строку "auto"
    init_bias: float | str = 2.88


@dataclass
class TrainConfig:
    loss_name: str = "wing"
    wing_w: float = 0.5  # <-- Добавлено
    wing_epsilon: float = 0.1  # <-- Добавлено

    batch_size: int = 32
    accum_steps: int = 1  # <-- Добавлено
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 7
    save_dir: str = "checkpoints"
    mixup_alpha: float = 0.0
    seed: int = 42

    use_ema: bool = False  # <-- Добавлено
    ema_decay: float = 0.999  # <-- Добавлено

    # --- Настройки SWA ---
    use_swa: bool = False
    swa_start: int = 35
    swa_lr: float = 5e-5

    # --- Настройки трекинга ---
    use_wandb: bool = True
    wandb_project: str = "cow_bcs"
    wandb_name: str = "dinov3_small_frozen_wing_balanced_softlabels"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(config_path: str = "config/train.yaml") -> Config:
    base_cfg = OmegaConf.structured(Config)
    path = Path(config_path)

    if not path.exists():
        print(f"Файл '{config_path}' не найден. Используются дефолты.")
        return base_cfg

    yaml_cfg = OmegaConf.load(path)
    merged_cfg = OmegaConf.merge(base_cfg, yaml_cfg)

    cli_cfg = OmegaConf.from_cli()
    merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)

    return cast(Config, merged_cfg)
