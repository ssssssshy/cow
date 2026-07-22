import timm
import torch
import torch.nn as nn


class CowBCSModel(nn.Module):
    """Модель прямой регрессии для оценки упитанности коров (BCS)."""

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        drop_rate: float = 0.3,
        init_bias: float = 2.88,  # Стартовое значение (средний BCS по датасету)
    ):
        super().__init__()
        self.model_name = model_name

        # Для прямой регрессии (Huber/Wing/MSE) нам нужен ровно 1 выход!
        self.num_outputs = 1

        # Создаем backbone из timm с настроенным числом выходов
        self.backbone: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_outputs,
            drop_rate=drop_rate,
        )

        # 🔥 Трюк для быстрого старта: инициализируем bias финального слоя
        # средним значением BCS. Модель с первого батча будет выдавать ~2.88
        classifier = self.backbone.get_classifier()  # type: ignore
        if isinstance(classifier, nn.Linear):
            nn.init.constant_(classifier.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает предсказание размера [Batch_size]."""
        out: torch.Tensor = self.backbone(x)

        # 🔥 Исправление ошибки broadcasting:
        # Превращаем тензор [Batch_size, 1] в плоский [Batch_size]
        return out.view(-1)


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование модели регрессии на устройстве: {device}")

    # Тестируем с EfficientNetV2-S
    model = CowBCSModel(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, init_bias=2.88
    ).to(device)

    dummy_input = torch.randn(4, 3, 384, 384).to(device)
    dummy_output = model(dummy_input)

    print(f"Формат входа:  {dummy_input.shape}")
    print(f"Формат выхода (предсказания): {dummy_output.shape}")  # Ожидается [4]
    print(
        f"Тестовые предсказания (должны быть около 2.88):\n{dummy_output.detach().cpu().numpy()}"
    )
