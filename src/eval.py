from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config import load_config
from src.data import get_dataloaders
from src.metrics import compute_all_metrics
from src.models import CowBCSModel


def load_model(cfg, model_path, device):
    """Загружает модель, очищая ключи DDP (префикс 'module.')."""
    model = CowBCSModel(
        model_name=cfg.model.name,
        pretrained=False,  # Веса мы загрузим свои
    ).to(device)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Очищаем префикс "module." от DistributedDataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        clean_state_dict[name] = v

    model.load_state_dict(clean_state_dict)
    model.eval()
    return model


@torch.inference_mode()
def get_predictions(model, loader, device):
    """Прогоняет датасет и собирает все предсказания."""
    all_preds = []
    all_targets = []

    for images, targets, _ in tqdm(loader, desc="Оценка модели"):
        images = images.to(device)
        preds = model(images)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_analysis(preds: np.ndarray, targets: np.ndarray, save_dir: Path):
    """Строит и сохраняет аналитические графики."""
    save_dir.mkdir(exist_ok=True, parents=True)

    # Общие настройки графиков
    sns.set_theme(style="whitegrid")

    # 1. Scatter Plot (Факт vs Предсказание)
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, color="blue", edgecolor="k")

    # Идеальная линия предсказания y = x
    min_val, max_val = 1.0, 5.0
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Идеальное предсказание",
    )

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Фактический BCS")
    plt.ylabel("Предсказанный BCS")
    plt.title("Scatter Plot: Предсказания модели vs Факт")
    plt.legend()
    plt.savefig(save_dir / "scatter_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Матрица ошибок (Confusion Matrix)
    # Округляем до ближайшего 0.25 для создания классов
    step = 0.25
    rounded_preds = np.round(preds / step) * step
    rounded_targets = np.round(targets / step) * step

    classes = np.arange(1.0, 5.25, 0.25)
    # Превращаем числа в красивые строки, например "2.75", для Seaborn
    str_classes = [f"{c:.2f}" for c in classes]

    cm = confusion_matrix(rounded_targets, rounded_preds, labels=classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=str_classes,
        yticklabels=str_classes,
    )
    plt.xlabel("Предсказанный класс (округленный)")
    plt.ylabel("Фактический класс")
    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Распределение ошибок
    errors = preds - targets
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=40, kde=True, color="purple")
    plt.axvline(x=0, color="r", linestyle="--", lw=2)
    plt.axvline(x=0.25, color="orange", linestyle=":", lw=2, label="+0.25 (Допуск)")
    plt.axvline(x=-0.25, color="orange", linestyle=":", lw=2, label="-0.25 (Допуск)")
    plt.xlabel("Ошибка (Предсказание - Факт)")
    plt.ylabel("Количество изображений")
    plt.title("Гистограмма распределения ошибок")
    plt.legend()
    plt.savefig(save_dir / "error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Графики успешно сохранены в папку: {save_dir}")


def main():
    cfg = load_config("config/train.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Запуск оценки на устройстве: {device}")

    model_path = Path(cfg.train.save_dir) / "best_bcs_model.pt"
    results_dir = Path("results")

    # 1. Загрузка данных (Используем только Val, DDP выключен!)
    _, val_loader = get_dataloaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.train.batch_size,
        img_size=(cfg.data.img_size[0], cfg.data.img_size[1]),
        crop_bbox=cfg.data.crop_bbox,
        is_distributed=False,
        num_workers=cfg.data.num_workers,
    )

    # 2. Загрузка модели
    model = load_model(cfg, model_path, device)

    # 3. Инференс
    preds, targets = get_predictions(model, val_loader, device)

    # 4. Расчет финальных метрик
    metrics = compute_all_metrics(torch.tensor(preds), torch.tensor(targets))
    print("\n📊 Финальные метрики на валидации:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")

    # 5. Отрисовка графиков
    plot_analysis(preds, targets, results_dir)


if __name__ == "__main__":
    main()
