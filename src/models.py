import timm
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Канальное внимание: решает 'ЧТО' важно на картинке."""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Общая полносвязная сеть для обоих пулингов
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Пространственное внимание: решает 'ГДЕ' находится важная информация."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Вычисляем среднее и максимум по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Соединяем их вместе
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Сборка модулей канального и пространственного внимания."""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Умножаем входной тензор на маски внимания
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class CowBCSModel(nn.Module):
    """Модель прямой регрессии BCS с модулем внимания CBAM."""

    def __init__(
        self,
        model_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        drop_rate: float = 0.5,
        init_bias: float = 2.88,
    ):
        super().__init__()
        self.model_name = model_name

        # 🔥 Трюк с timm:
        # num_classes=0 отсекает финальный классификатор
        # global_pool='' отсекает усреднение, заставляя модель возвращать
        # пространственный тензор размерностью [B, C, H, W]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            drop_path_rate=0.2,  # Stochastic Depth для ConvNeXt
        )

        # Получаем количество выходных каналов (для convnext_small это 768)
        self.num_features = int(getattr(self.backbone, "num_features"))

        # Инициализируем наш модуль CBAM
        self.cbam = CBAM(self.num_features, ratio=16, kernel_size=7)

        # Финальные слои: пулинг -> дропаут -> предсказание одного числа
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, 1)

        # Инициализация смещения (Bias) для быстрого старта
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Извлекаем признаки (карта [B, C, H, W])
        features = self.backbone(x)

        # 2. Накладываем тепловую карту CBAM, которая подавит фон
        attended_features = self.cbam(features)

        # 3. Сворачиваем пространственные размерности в вектор [B, C, 1, 1] и плющим [B, C]
        pooled = self.global_pool(attended_features).flatten(1)

        # 4. Регрессия
        pooled = self.dropout(pooled)
        out = self.head(pooled)

        return out.view(-1)


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование модели CBAM на устройстве: {device}")

    # Тестируем с ConvNeXt
    model = CowBCSModel(
        model_name="convnext_small.fb_in22k_ft_in1k_384", pretrained=False
    ).to(device)

    # Имитируем батч из 4 картинок
    dummy_input = torch.randn(4, 3, 384, 384).to(device)
    dummy_output = model(dummy_input)

    print(f"Формат входа:  {dummy_input.shape}")
    print(f"Формат выхода: {dummy_output.shape}")
