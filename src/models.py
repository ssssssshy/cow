import timm
import torch
import torch.nn as nn


class CowBCSModel(nn.Module):
    """Модель порядковой регрессии (Ordinal Regression) для оценки упитанности коров (BCS)."""

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        drop_rate: float = 0.3,
        num_classes: int = 17,  # Всего 17 классов для шкалы 1.0 - 5.0 с шагом 0.25
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Для порядковой регрессии количество выходов равно (num_classes - 1)
        self.num_outputs = num_classes - 1

        # Создаем backbone из timm с настроенным числом выходов
        self.backbone: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_outputs,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает сырые логиты размера [Batch_size, num_classes - 1] для Ordinal Loss."""
        out: torch.Tensor = self.backbone(x)
        return out


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование модели порядковой регрессии на устройстве: {device}")

    # Тестируем с EfficientNetV2-S
    model = CowBCSModel(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, num_classes=17
    ).to(device)

    dummy_input = torch.randn(4, 3, 384, 384).to(device)
    dummy_output = model(dummy_input)

    print(f"Формат входа:  {dummy_input.shape}")
    print(f"Формат выхода (логиты порогов): {dummy_output.shape}")  # Ожидается [4, 16]
    print(f"Тестовые предсказания:\n{dummy_output.detach().cpu().numpy()}")
