import timm
import torch
import torch.nn as nn


class CowBCSModel(nn.Module):
    """Модель регрессии для оценки упитанности коров (BCS)."""

    def __init__(
        self,
        model_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        init_bias: float = 2.8,
    ):
        super().__init__()
        self.model_name = model_name

        # Создаем backbone из timm
        self.backbone: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=drop_rate,
        )

        # Начальная инициализация bias под среднее BCS
        self._init_bias(init_bias)

    def _init_bias(self, init_value: float) -> None:
        """Находит классификатор и задает начальный bias."""
        head = self.backbone.get_classifier()  # type: ignore

        if isinstance(head, nn.Linear) and head.bias is not None:
            nn.init.constant_(head.bias, init_value)
        else:
            for module in reversed(list(self.backbone.modules())):
                if (
                    isinstance(module, nn.Linear)
                    and module.out_features == 1
                    and module.bias is not None
                ):
                    nn.init.constant_(module.bias, init_value)
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Важно: используем отдельную переменную out, не перезаписывая self.backbone
        out: torch.Tensor = self.backbone(x)
        return out.view(-1)


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование модели на устройстве: {device}")

    model = CowBCSModel(
        model_name="convnext_small.fb_in22k_ft_in1k_384", pretrained=False
    ).to(device)

    dummy_input = torch.randn(4, 3, 384, 384).to(device)
    dummy_output = model(dummy_input)

    print(f"Формат входа:  {dummy_input.shape}")
    print(f"Формат выхода: {dummy_output.shape}")
    print(f"Тестовые предсказания: {dummy_output.detach().cpu().numpy()}")
