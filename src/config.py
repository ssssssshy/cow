from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    img_size: tuple[int, int] = (384, 384)
    crop_bbox: bool = True

    # Параметры для жесткого кропа
    margin_left: float = 0.05
    margin_right: float = 0.05
    margin_top: float = 0.0
    margin_bottom: float = 0.30

    bbox_pad: float = 0.05
    num_workers: int = 4
    target_noise: float = 0.05

    use_soft_labels: bool = False
    soft_label_sigma: float = 0.25
    use_balanced_sampler: bool = True
    class_weight_beta: float = 0.999


@dataclass
class ModelConfig:
    name: str = "convnext_tiny"
    pretrained: bool = True
    freeze_backbone: bool = False
    use_cls_token: bool = True
    use_patch_tokens: bool = False
    patch_pool: str = "avg"
    drop_rate: float = 0.3
    init_bias: float | str = 2.88


@dataclass
class TrainConfig:
    loss_name: str = "wing"
    wing_w: float = 0.5
    wing_epsilon: float = 0.1

    batch_size: int = 32
    accum_steps: int = 1
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 7
    save_dir: str = "checkpoints"
    mixup_alpha: float = 0.0
    seed: int = 42

    use_ema: bool = False
    ema_decay: float = 0.999

    # Настройки SWA
    use_swa: bool = False
    swa_start: int = 35
    swa_lr: float = 5e-5

    # Настройки трекинга
    use_wandb: bool = True
    wandb_project: str = "cow_bcs"
    wandb_name: str = "convnext_wing_tight_crop"


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
