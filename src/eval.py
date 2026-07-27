from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from models import CowBCSModel
from src.config import load_config

# Обязательно импортируйте get_test_dataloader из вашего data.py
from src.data import get_test_dataloader
from src.metrics import compute_all_metrics
from src.utils import set_seed


def load_model(cfg, model_path, device):
    """Загружает модель, используя настройки из конфига, и очищает ключи DDP."""
    # Используем cfg.model вместо жестко заданных параметров
    model = CowBCSModel(cfg.model, img_size=tuple(cfg.data.img_size)).to(device)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.removeprefix("module.")
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
        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)

        all_preds.extend(preds.view(-1).cpu().numpy())
        all_targets.extend(targets.view(-1).cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_analysis(preds: np.ndarray, targets: np.ndarray, save_dir: Path):
    """Строит и сохраняет аналитические графики."""
    save_dir.mkdir(exist_ok=True, parents=True)
    sns.set_theme(style="whitegrid")

    # 1. Scatter Plot (Факт vs Предсказание)
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, color="blue", edgecolor="k")

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
    def bcs_to_idx(arr):
        return np.clip(np.round((arr - 1.0) / 0.25), 0, 16).astype(int)

    int_preds = bcs_to_idx(preds)
    int_targets = bcs_to_idx(targets)

    num_classes = 17
    labels = np.arange(num_classes)
    str_classes = [f"{(1.0 + i * 0.25):.2f}" for i in range(num_classes)]

    cm = confusion_matrix(int_targets, int_preds, labels=labels)

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

    print(f"Графики успешно сохранены в папку: {save_dir}")


def mine_and_plot_hard_examples(
    preds: np.ndarray, targets: np.ndarray, dataset, save_dir: Path, top_k=50
):
    """Ищет примеры с наибольшей ошибкой, сохраняет их в CSV и рисует ТОП-16."""
    errors = np.abs(preds - targets)

    results = []
    for idx in range(len(errors)):
        sample_meta = dataset.samples[idx]
        results.append(
            {
                "image_path": str(sample_meta["img_path"]),
                "class_id": sample_meta["class_id"],
                "true_bcs": targets[idx],
                "pred_bcs": preds[idx],
                "error": errors[idx],
            }
        )

    df_hard = pd.DataFrame(results).sort_values(by="error", ascending=False).head(top_k)

    csv_path = save_dir / "hard_examples_report.csv"
    df_hard.to_csv(csv_path, index=False)
    print(f"Отчет по ТОП-{top_k} сложным примерам сохранен в: {csv_path}")

    num_to_plot = min(16, len(df_hard))
    if num_to_plot == 0:
        return

    cols = 4
    rows = (num_to_plot + cols - 1) // cols
    _fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i in range(num_to_plot):
        row = df_hard.iloc[i]
        img_path = row["image_path"]

        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[i].imshow(image)
        else:
            axes[i].text(0.5, 0.5, "Image not found", ha="center", va="center")

        axes[i].set_title(
            f"True: {row['true_bcs']:.2f} | Pred: {row['pred_bcs']:.2f}\nErr: {row['error']:.2f}",
            color="red" if row["error"] > 0.5 else "orange",
            fontsize=10,
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plot_path = save_dir / "hard_examples_top16.png"
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сетка проблемных изображений сохранена в: {plot_path}")


def main():
    cfg = load_config("config/train.yaml")
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск оценки на устройстве: {device}")

    model_path = Path(cfg.train.save_dir) / "best_bcs_model.pt"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # 1. Загрузка данных (Используем get_test_dataloader)
    test_loader = get_test_dataloader(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.train.batch_size,
        img_size=tuple(cfg.data.img_size),
        crop_bbox=cfg.data.crop_bbox,
        num_workers=cfg.data.num_workers,
        margin_left=cfg.data.margin_left,
        margin_right=cfg.data.margin_right,
        margin_top=cfg.data.margin_top,
        margin_bottom=cfg.data.margin_bottom,
    )

    # 2. Загрузка модели
    model = load_model(cfg, model_path, device)

    # 3. Инференс
    preds, targets = get_predictions(model, test_loader, device)

    # 4. Расчет финальных метрик
    metrics = compute_all_metrics(torch.tensor(preds), torch.tensor(targets))
    print("\nФинальные метрики на тесте:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")

    # 5. Отрисовка базовых графиков
    plot_analysis(preds, targets, results_dir)

    # 6. Анализ "Hard Examples"
    mine_and_plot_hard_examples(
        preds, targets, test_loader.dataset, results_dir, top_k=50
    )


if __name__ == "__main__":
    main()
