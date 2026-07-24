import timm
import torch
from torch import nn


class CowBCSModel(nn.Module):
    """
    Чистая модель прямой регрессии для оценки упитанности коров (BCS).
    Использует базовую архитектуру без дополнительных модулей внимания.
    """

    def __init__(
        self,
        model_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        drop_rate: float = 0.4,
        init_bias: float = 2.88,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_outputs = 1

        # Создаем чистый backbone из timm
        self.backbone: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_outputs,
            drop_rate=drop_rate,
            drop_path_rate=0.2,  # Защита от переобучения (Stochastic Depth)
        )

        # Трюк для быстрого старта: инициализируем bias финального слоя
        classifier = self.backbone.get_classifier()  # type: ignore
        if isinstance(classifier, nn.Linear):
            nn.init.constant_(classifier.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает предсказание размера [Batch_size]."""
        out: torch.Tensor = self.backbone(x)
        return out.view(-1)


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование чистой модели на устройстве: {device}")

    model = CowBCSModel(
        model_name="convnext_small.fb_in22k_ft_in1k_384",
        pretrained=False,
        init_bias=2.88,
    ).to(device)

    dummy_input = torch.randn(4, 3, 384, 384).to(device)
    dummy_output = model(dummy_input)

    print(f"Формат входа:  {dummy_input.shape}")
    print(f"Формат выхода (предсказания): {dummy_output.shape}")
