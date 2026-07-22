from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    """Wing Loss для точной регрессии непрерывных параметров.

    Обеспечивает более сильные градиенты при малых ошибках, стимулируя высокую
    точность.
    """

    def __init__(
        self, omega: float = 10.0, epsilon: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        self.c = omega - omega * torch.log(torch.tensor(1.0 + omega / epsilon))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(preds - targets)
        c = self.c.to(preds.device)

        # Переходная формула Wing Loss
        flag = delta < self.omega
        loss_small = self.omega * torch.log(1.0 + delta / self.epsilon)
        loss_large = delta - c

        loss = torch.where(flag, loss_small, loss_large)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 Loss с возможностью взвешивания примеров в зависимости от
    редкости класса."""

    def __init__(self, beta: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.smooth_l1(preds, targets)

        if sample_weights is not None:
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Cross-Entropy Loss для задачи BCS.
    Превращает непрерывную/порядковую задачу в K-1 бинарных классификаций.
    """

    def __init__(self, num_classes: int = 17):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds: сырые логиты от модели с размером [Batch_size, num_classes - 1]
        targets: индексы классов от 0 до num_classes - 1 (в виде тензора целых чисел)
        """
        device = preds.device

        # Создаем матрицу бинарных таргетов для каждого порога
        # Например, если таргет класса 8, то для порогов 0..7 таргет = 1, а для 8..15 таргет = 0
        levels = torch.arange(self.num_classes - 1, device=device).view(
            1, -1
        )  # [1, K-1]
        target_levels = (targets.view(-1, 1) > levels).float()  # [Batch_size, K-1]

        # Считаем бинарную кросс-энтропию с логитами для каждого порога
        loss = F.binary_cross_entropy_with_logits(
            preds, target_levels, reduction="none"
        )

        # Суммируем потери по всем порогам и усредняем по батчу
        return loss.sum(dim=1).mean()


def get_loss_function(loss_name: str = "smooth_l1", beta: float = 0.1) -> nn.Module:
    """Фабрика для удобного выбора Loss-функции."""
    loss_name = loss_name.lower()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=beta)
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "wing":
        return WingLoss()
    elif loss_name == "weighted_smooth_l1":
        return WeightedSmoothL1Loss(beta=beta)
    else:
        raise ValueError(f"Неизвестная функция потерь: {loss_name}")


# --- Quick Test ---
if __name__ == "__main__":
    preds = torch.tensor([2.80, 2.90, 2.45, 3.60])
    targets = torch.tensor([2.75, 3.00, 2.50, 3.25])

    smooth_l1 = get_loss_function("smooth_l1")
    wing = get_loss_function("wing")

    print(f"Smooth L1 Loss : {smooth_l1(preds, targets).item():.4f}")
    print(f"Wing Loss      : {wing(preds, targets).item():.4f}")
