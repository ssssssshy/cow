from dataclasses import dataclass

import timm
import torch
from torch import nn


@dataclass
class ModelConfig:
    """Временный конфиг для тестирования (в реальном коде импортируйте из config.py)"""

    name: str = "vit_small_patch16_dinov3.lvd1689m"
    pretrained: bool = True
    freeze_backbone: bool = False
    use_cls_token: bool = True
    use_patch_tokens: bool = True
    patch_pool: str = "avg"  # "avg" или "max"
    drop_rate: float = 0.3
    init_bias: float | str = 2.88


class CowBCSModel(nn.Module):
    """
    Гибкая модель для BCS с поддержкой:
    - Заморозки backbone
    - Кастомного пулинга (CLS + patches для ViT, GAP для CNN)
    - Разных архитектур (ViT, ConvNeXt, EfficientNet)
    """

    def __init__(self, cfg: ModelConfig, img_size: tuple[int, int] = (384, 384)):
        super().__init__()
        self.cfg = cfg
        self.freeze_backbone = cfg.freeze_backbone

        # Определяем, является ли модель ViT (для кастомного пулинга)
        self.is_vit = "vit" in cfg.name.lower() or "deit" in cfg.name.lower()

        # Создаем backbone с явным типом для Pylance
        self.backbone: nn.Module = timm.create_model(
            cfg.name,
            pretrained=cfg.pretrained,
            num_classes=0,  # Убираем классификатор timm
            drop_rate=cfg.drop_rate,
            drop_path_rate=0.1,
        )

        # Заморозка backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Переключаем в eval mode для отключения Dropout/DropPath
            self.backbone.eval()

        # Вычисляем размер фич на dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
            # Явно вызываем forward_features, а не __call__
            dummy_features = self.backbone.forward_features(dummy_input)

        # Определяем in_features в зависимости от архитектуры
        self.global_pool: nn.Module

        if self.is_vit:
            # ViT возвращает [B, N+1, D] где N+1 = CLS + patches
            if dummy_features.ndim == 3:
                # Shape: [B, num_tokens, embed_dim]
                embed_dim = dummy_features.shape[-1]

                if cfg.use_cls_token and cfg.use_patch_tokens:
                    # CLS + mean/max patches -> concat -> Linear
                    self.in_features = embed_dim * 2
                elif cfg.use_cls_token or cfg.use_patch_tokens:
                    self.in_features = embed_dim
                else:
                    raise ValueError(
                        "Должен быть включен хотя бы один: cls_token или patch_tokens"
                    )
            else:
                # Fallback: если timm вернул уже pooled features [B, D]
                self.in_features = dummy_features.shape[-1]
                self.is_vit = False  # Значит это не ViT или timm сам сделал pooling

        else:
            # CNN (ConvNeXt, EfficientNet) возвращает [B, C, H, W]
            # Нужен global average pooling
            if dummy_features.ndim == 4:
                self.in_features = dummy_features.shape[1]  # Channels
                self.global_pool = nn.AdaptiveAvgPool2d(1)
            else:
                # Уже pooled [B, C]
                self.in_features = dummy_features.shape[-1]
                self.global_pool = nn.Identity()

        # Создаем голову
        self.head = nn.Linear(self.in_features, 1)

        # Инициализация bias
        if isinstance(cfg.init_bias, (int, float)):
            nn.init.constant_(self.head.bias, float(cfg.init_bias))
        elif cfg.init_bias == "auto":
            # Для "auto" нужно вычислить mean BCS из трейна
            # Здесь можно оставить 2.88 как дефолт или передать из train.py
            nn.init.constant_(self.head.bias, 2.88)
            print("⚠️ init_bias='auto' требует вычисления mean BCS из train dataset")

        # Инициализация weights
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass с кастомным пулингом для ViT.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Predictions [B]
        """
        # Если backbone заморожен, отключаем градиенты для ускорения
        with torch.set_grad_enabled(not self.freeze_backbone):
            # Явно вызываем forward_features вместо __call__
            features = self.backbone.forward_features(x)

        # Пулинг в зависимости от архитектуры
        if self.is_vit and features.ndim == 3:
            # ViT: features shape [B, N+1, D]
            # Первый токен - CLS, остальные - patches
            cls_token = features[:, 0, :]  # [B, D]
            patch_tokens = features[:, 1:, :]  # [B, N, D]

            # Пулинг patch tokens
            if self.cfg.patch_pool == "avg":
                pooled_patches = patch_tokens.mean(dim=1)  # [B, D]
            elif self.cfg.patch_pool == "max":
                pooled_patches = patch_tokens.max(dim=1)[0]  # [B, D]
            else:
                raise ValueError(f"Неизвестный patch_pool: {self.cfg.patch_pool}")

            # Объединение
            if self.cfg.use_cls_token and self.cfg.use_patch_tokens:
                pooled_features = torch.cat(
                    [cls_token, pooled_patches], dim=1
                )  # [B, 2*D]
            elif self.cfg.use_cls_token:
                pooled_features = cls_token  # [B, D]
            elif self.cfg.use_patch_tokens:
                pooled_features = pooled_patches  # [B, D]
            else:
                raise ValueError(
                    "Должен быть включен хотя бы один: cls_token или patch_tokens"
                )

        else:
            # CNN или уже pooled ViT
            if features.ndim == 4:
                # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
                pooled_features = self.global_pool(features).squeeze(-1).squeeze(-1)
            else:
                # Уже [B, C]
                pooled_features = features

        # Предсказание
        output = self.head(pooled_features)  # [B, 1]
        return output.squeeze(-1)  # [B]

    def train(self, mode: bool = True) -> "CowBCSModel":
        """
        Переопределяем train() чтобы замороженный backbone оставался в eval mode.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Тестирование модели на устройстве: {device}\n")

    # Тест 1: ViT с frozen backbone
    print("=" * 70)
    print("Test 1: DINOv3 ViT-Small (frozen backbone + CLS + patches)")
    print("=" * 70)

    cfg1 = ModelConfig(
        name="vit_small_patch16_dinov3.lvd1689m",
        pretrained=False,  # Для быстрого теста
        freeze_backbone=True,
        use_cls_token=True,
        use_patch_tokens=True,
        patch_pool="avg",
        drop_rate=0.3,
        init_bias=2.88,
    )

    model1 = CowBCSModel(cfg1, img_size=(384, 384)).to(device)

    dummy_input = torch.randn(4, 3, 384, 384).to(device)

    # Проверяем, что backbone заморожен
    backbone_params_trainable = sum(
        p.requires_grad for p in model1.backbone.parameters()
    )
    head_params_trainable = sum(p.requires_grad for p in model1.head.parameters())

    print(f"Формат входа: {dummy_input.shape}")

    with torch.no_grad():
        output1 = model1(dummy_input)
    print(f"Формат выхода: {output1.shape}")
    print(f"Backbone trainable params: {backbone_params_trainable} (должно быть 0)")
    print(f"Head trainable params: {head_params_trainable} (должно быть > 0)")
    print(f"in_features: {model1.in_features}")
    print()

    # Тест 2: ConvNeXt (CNN)
    print("=" * 70)
    print("Test 2: ConvNeXt Small (trainable backbone)")
    print("=" * 70)

    cfg2 = ModelConfig(
        name="convnext_small.fb_in22k_ft_in1k_384",
        pretrained=False,
        freeze_backbone=False,
        use_cls_token=False,
        use_patch_tokens=False,
        drop_rate=0.2,
        init_bias=3.0,
    )

    model2 = CowBCSModel(cfg2, img_size=(384, 384)).to(device)

    backbone_params_trainable2 = sum(
        p.requires_grad for p in model2.backbone.parameters()
    )
    head_params_trainable2 = sum(p.requires_grad for p in model2.head.parameters())

    print(f"Формат входа: {dummy_input.shape}")

    with torch.no_grad():
        output2 = model2(dummy_input)
    print(f"Формат выхода: {output2.shape}")
    print(f"Backbone trainable params: {backbone_params_trainable2} (должно быть > 0)")
    print(f"Head trainable params: {head_params_trainable2}")
    print(f"in_features: {model2.in_features}")
    print()

    # Тест 3: ViT только с CLS token
    print("=" * 70)
    print("Test 3: DINOv3 ViT-Small (только CLS token)")
    print("=" * 70)

    cfg3 = ModelConfig(
        name="vit_small_patch16_dinov3.lvd1689m",
        pretrained=False,
        freeze_backbone=False,
        use_cls_token=True,
        use_patch_tokens=False,
        drop_rate=0.3,
        init_bias=2.88,
    )

    model3 = CowBCSModel(cfg3, img_size=(384, 384)).to(device)

    print(f"in_features: {model3.in_features} (должно быть 384 для ViT-Small)")

    with torch.no_grad():
        output3 = model3(dummy_input)
    print(f"Формат выхода: {output3.shape}")

    print("\n✅ Все тесты пройдены успешно!")
