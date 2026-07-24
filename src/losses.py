import torch
import torch.nn.functional as F
from torch import nn


class WingLoss(nn.Module):
    """
    Wing Loss, адаптированный под узкий диапазон (BCS от 1.0 до 5.0).
    """

    def __init__(
        self, omega: float = 0.5, epsilon: float = 0.1, reduction: str = "mean"
    ):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        # Использование float тензора для вычисления C
        self.c = omega - omega * torch.log(
            torch.tensor(1.0 + omega / epsilon, dtype=torch.float32)
        )

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
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.smooth_l1(preds, targets)

        if sample_weights is not None:
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HuberLossWithDelta(nn.Module):
    """
    Huber Loss для регрессии, устойчивый к выбросам (шуму).
    Комбинирует L1 (для больших ошибок) и L2 (для малых ошибок).
    Использование малой дельты (например, 0.05) помогает игнорировать
    значительные шумы в разметке.
    """

    def __init__(self, delta: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Используем встроенную функцию PyTorch для Huber Loss
        return F.huber_loss(preds, targets, reduction=self.reduction, delta=self.delta)


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


class BCSMagneticLoss(nn.Module):
    """
    Гибридная функция потерь специально для индекса BCS.
    Объединяет базовую регрессию (Huber) с "магнитной" сеткой,
    которая штрафует предсказания, зависающие между шагами 0.25.
    """

    def __init__(
        self, delta: float = 0.05, mag_weight: float = 0.1, step: float = 0.25
    ):
        super().__init__()
        self.delta = delta  # Порог Huber
        self.mag_weight = mag_weight  # Сила "магнита" (штрафа за дробные значения)
        self.step = step  # Шаг шкалы BCS (0.25)

        # Коэффициент для перевода шага 0.25 в период синуса (Пи)
        # Если step = 0.25, то multiplier = 4.0
        self.multiplier = 1.0 / step

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Базовая стабильная регрессия (чтобы модель поняла общую упитанность коровы)
        huber_loss = F.huber_loss(preds, targets, reduction="mean", delta=self.delta)

        # 2. Магнитный штраф (Quantization Penalty)
        # Умножаем предсказание на 4 и на Пи.
        # Если pred = 2.75  -> sin(11 * Pi) = 0 (Штрафа нет, мы попали в сетку)
        # Если pred = 2.875 -> sin(11.5 * Pi) = 1 (Максимальный штраф, мы застряли посередине)
        magnetic_penalty = torch.sin(preds * self.multiplier * torch.pi) ** 2
        magnetic_loss = magnetic_penalty.mean() * self.mag_weight

        # Итоговый лосс: Тянем к правильному ответу + заставляем встать ровно на шаг 0.25
        return huber_loss + magnetic_loss


def get_loss_function(
    loss_name: str = "smooth_l1",
    beta: float = 0.1,
    huber_delta: float = 0.05,
    wing_omega: float = 0.5,
    wing_epsilon: float = 0.1,
    mag_weight: float = 0.05,  # Добавляем силу магнита
) -> nn.Module:
    """Фабрика для удобного выбора Loss-функции."""
    loss_name = loss_name.lower()

    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=beta)
    elif loss_name == "huber":
        return HuberLossWithDelta(delta=huber_delta)
    elif loss_name == "magnetic":  # 🔥 Наш новый гибрид
        return BCSMagneticLoss(delta=huber_delta, mag_weight=mag_weight)
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "wing":
        return WingLoss(omega=wing_omega, epsilon=wing_epsilon)
    elif loss_name == "weighted_smooth_l1":
        return WeightedSmoothL1Loss(beta=beta)
    elif loss_name == "ordinal":
        return OrdinalRegressionLoss()
    else:
        raise ValueError(f"Неизвестная функция потерь: {loss_name}")


# --- Quick Test ---
if __name__ == "__main__":
    preds = torch.tensor([2.80, 2.90, 2.45, 3.60])
    targets = torch.tensor([2.75, 3.00, 2.50, 3.25])

    smooth_l1 = get_loss_function("smooth_l1")
    wing = get_loss_function("wing")
    huber = get_loss_function("huber", huber_delta=0.05)

    print(f"Smooth L1 Loss : {smooth_l1(preds, targets).item():.4f}")
    print(f"Wing Loss      : {wing(preds, targets).item():.4f}")
    print(f"Huber Loss     : {huber(preds, targets).item():.4f}")
