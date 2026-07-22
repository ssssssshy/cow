from dataclasses import dataclass, field
from pathlib import Path
from typing import List, cast

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    img_size: List[int] = field(default_factory=lambda: [384, 384])
    crop_bbox: bool = True
    num_workers: int = 4
    target_noise: float = 0.05  # Интенсивность зашумления меток для защиты от шума


@dataclass
class ModelConfig:
    name: str = "convnext_small.fb_in22k_ft_in1k_384"
    pretrained: bool = True
    drop_rate: float = 0.2
    init_bias: float = 2.8


@dataclass
class TrainConfig:
    loss_name: str = "smooth_l1"
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    patience: int = 7
    save_dir: str = "checkpoints"
    mixup_alpha: float = 0.0

    # --- Настройки SWA (Stochastic Weight Averaging) ---
    use_swa: bool = True
    swa_start: int = 35
    swa_lr: float = 5e-5

    # --- Настройки трекинга ---
    use_wandb: bool = False
    wandb_project: str = "cow_bcs"
    wandb_name: str = "run_01_baseline"


@dataclass
class Config:
    """Главный конфигурационный класс."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(config_path: str = "config/train.yaml") -> Config:
    """
    Загружает конфигурацию из YAML и мержит её с дефолтными значениями (Dataclasses).
    """
    # 1. Создаем базовый конфиг из схемы (со всеми типами и дефолтами)
    base_cfg = OmegaConf.structured(Config)

    path = Path(config_path)
    if not path.exists():
        print(
            f"⚠️ Файл конфигурации '{config_path}' не найден. Используются значения по умолчанию."
        )
        return base_cfg

    # 2. Читаем YAML
    yaml_cfg = OmegaConf.load(path)
    merged_cfg = OmegaConf.merge(base_cfg, yaml_cfg)

    # Считываем аргументы из терминала (Kaggle) и применяем их поверх YAML
    cli_cfg = OmegaConf.from_cli()
    merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)

    return cast(Config, merged_cfg)


# --- Quick Test ---
if __name__ == "__main__":
    cfg = load_config("config/train.yaml")
    print(OmegaConf.to_yaml(cfg))
