from src.config import load_config
from src.trainer import run_training


def main():
    # Загружаем конфигурацию (слияние dataclasses + yaml)
    cfg = load_config("config/train.yaml")

    # Запускаем оркестратор обучения
    run_training(cfg)


if __name__ == "__main__":
    main()
