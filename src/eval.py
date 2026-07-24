from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config import load_config
from src.data import get_dataloaders
from src.metrics import compute_all_metrics
from src.models import CowBCSModel
from src.utils import set_seed


def load_model(cfg, model_path, device):
    """Загружает модель, очищая ключи DDP (префикс 'module.')."""
    model = CowBCSModel(
        model_name=cfg.model.name,
        pretrained=False,
        init_bias=cfg.model.init_bias,  # Не забываем про bias
    ).to(device)

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

        # Обязательно "плющим" тензоры, чтобы они точно были 1D массивами
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
    # Превращаем float-значения (1.0, 1.25 ...) в целые индексы от 0 до 16,
    # так как sklearn категорически не принимает float для confusion_matrix.
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

    print(f"✅ Графики успешно сохранены в папку: {save_dir}")


def mine_and_plot_hard_examples(
    preds: np.ndarray, targets: np.ndarray, dataset, save_dir: Path, top_k=50
):
    """Ищет примеры с наибольшей ошибкой, сохраняет их в CSV и рисует ТОП-16."""
    errors = np.abs(preds - targets)

    results = []
    for idx in range(len(errors)):
        # Берем метаданные из оригинального датасета (путь к картинке и т.д.)
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

    # Сортируем по убыванию ошибки
    df_hard = pd.DataFrame(results).sort_values(by="error", ascending=False).head(top_k)

    # 1. Сохраняем в CSV
    csv_path = save_dir / "hard_examples_report.csv"
    df_hard.to_csv(csv_path, index=False)
    print(f"✅ Отчет по ТОП-{top_k} сложным примерам сохранен в: {csv_path}")

    # 2. Рисуем сетку картинок (ТОП-16)
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
    print(f"📸 Сетка проблемных изображений сохранена в: {plot_path}")


def main():
    cfg = load_config("config/train.yaml")
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Запуск оценки на устройстве: {device}")

    model_path = Path(cfg.train.save_dir) / "best_bcs_model.pt"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # 1. Загрузка данных (Используем только Val, DDP выключен! Shuffle по умолчанию выключен)
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

    # 5. Отрисовка базовых графиков (Матрица ошибок, гистограмма)
    plot_analysis(preds, targets, results_dir)

    # 6. Анализ "Hard Examples" (Майнинг сложных примеров)
    # Передаем val_loader.dataset, чтобы извлечь пути к оригинальным изображениям
    mine_and_plot_hard_examples(
        preds, targets, val_loader.dataset, results_dir, top_k=50
    )


if __name__ == "__main__":
    main()
