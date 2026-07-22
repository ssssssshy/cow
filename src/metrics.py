from typing import Dict
import torch


def compute_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error (Средняя абсолютная ошибка в баллах BCS)."""
    return torch.abs(preds - targets).mean().item()


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Squared Error (Корень из среднеквадратичной ошибки)."""
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def compute_accuracy_within_tolerance(
    preds: torch.Tensor, targets: torch.Tensor, tolerance: float = 0.25
) -> float:
    """Процент предсказаний, попавших в допустимую погрешность (tolerance).

    Args:
        tolerance: 0.25 означает точность попадания в соседний класс (±0.25).
    """
    diff = torch.abs(preds - targets)
    correct = (diff <= tolerance).float().sum()
    return (correct / len(targets)).item() * 100.0


def compute_exact_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, step: float = 0.25
) -> float:
    """Точность попадания класс-в-класс после округления до ближайшего шага шкалы (0.25)."""
    rounded_preds = torch.round(preds / step) * step
    rounded_targets = torch.round(targets / step) * step
    correct = (rounded_preds == rounded_targets).float().sum()
    return (correct / len(targets)).item() * 100.0


def compute_all_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Считает полный комплекс метрик для оценки модели.

    Returns:
        Словарь с метриками: MAE, RMSE, Acc_exact, Acc_0.25, Acc_0.50
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    return {
        "mae": compute_mae(preds, targets),
        "rmse": compute_rmse(preds, targets),
        "acc_exact": compute_exact_accuracy(preds, targets),
        "acc_tol_0.25": compute_accuracy_within_tolerance(
            preds, targets, tolerance=0.25
        ),
        "acc_tol_0.50": compute_accuracy_within_tolerance(
            preds, targets, tolerance=0.50
        ),
    }


# --- Quick Test ---
if __name__ == "__main__":
    # Тестовые данные: Факт [2.75, 3.00, 2.50, 3.25]
    dummy_targets = torch.tensor([2.75, 3.00, 2.50, 3.25])
    # Предсказания модели: [2.80, 2.90, 2.45, 3.60]
    dummy_preds = torch.tensor([2.80, 2.90, 2.45, 3.60])

    metrics = compute_all_metrics(dummy_preds, dummy_targets)
    print("📊 Результаты расчета метрик:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.3f}")
