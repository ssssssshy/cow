import timm
import torch
from torch import nn


class RobustAttention(nn.Module):
    """
    Безопасный блок внимания (Channel Attention) с Residual Connection
    и Zero-Init LayerScale для стабильного дообучения поверх предобученных сетей.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        # 1. Нормализация входа (защита от резких выбросов активаций)
        self.norm = nn.BatchNorm2d(in_channels)

        # Базовый механизм Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 2. Трюк LayerScale (Zero-Init)
        # Начинаем с 0. На первой эпохе блок возвращает чистый residual.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Нормализуем фичи перед вычислением весов внимания
        norm_x = self.norm(x)
        attn = self.se(norm_x)

        # 3. Residual Connection + Умножение на gamma
        # Изначально gamma=0, поэтому возвращается просто x
        return residual + self.gamma * (x * attn)


class CowBCSModel(nn.Module):
    """Модель с интегрированным механизмом внимания и защитой от взрыва градиентов."""

    def __init__(
        self,
        model_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        drop_rate: float = 0.4,
        init_bias: float = 2.88,
    ):
        super().__init__()
        self.model_name = model_name

        # Создаем backbone из timm БЕЗ пулинга и финальной головы (получаем 4D тензор)
        self.backbone: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Убираем стандартный классификатор
            global_pool="",  # Сохраняем пространственную структуру [B, C, H, W]
            drop_rate=drop_rate,
            drop_path_rate=0.2,
        )

        in_features = self.backbone.num_features

        # Вставляем наш безопасный блок внимания
        self.attention = RobustAttention(in_channels=in_features)  # type: ignore

        # Собираем кастомную "голову"
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(in_features, 1)  # type: ignore

        # Трюк для быстрого старта: инициализируем bias
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает предсказание размера [Batch_size]."""
        # 1. Извлекаем сырые признаки [B, C, H, W]
        features = self.backbone(x)

        # 2. Применяем внимание
        attended_features = self.attention(features)

        # 3. Пулинг и регрессия
        pooled = self.flatten(self.pool(attended_features))
        out = self.head(pooled)

        return out.view(-1)


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование модели на устройстве: {device}")

    model = CowBCSModel(
        model_name="convnext_small.fb_in22k_ft_in1k_384",
        pretrained=False,
    ).to(device)

    dummy_input = torch.randn(4, 3, 384, 384).to(device)
    dummy_output = model(dummy_input)

    print(f"Формат входа:  {dummy_input.shape}")
    print(f"Формат выхода: {dummy_output.shape}")
