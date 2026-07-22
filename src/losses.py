from typing import Optional
import torch
import torch.nn as nn


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
