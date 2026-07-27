import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.config import ModelConfig
from src.models import CowBCSModel


class RegressionTarget:
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output


def load_image_tight_crop(
    image_path: str, size: int = 384
) -> tuple[np.ndarray, torch.Tensor, Image.Image]:
    img = Image.open(image_path).convert("RGB")

    label_path_1 = os.path.splitext(image_path)[0] + ".txt"
    label_path_2 = (
        os.path.splitext(image_path.replace("/images/", "/labels/"))[0] + ".txt"
    )

    label_path = None
    if os.path.exists(label_path_1):
        label_path = label_path_1
    elif os.path.exists(label_path_2):
        label_path = label_path_2

    if label_path:
        with open(label_path, "r") as f:
            lines = f.readlines()
            if lines:
                line = lines[0].strip().split()
                if len(line) >= 5:
                    _, x_c, y_c, w, h = map(float, line[:5])

                    img_w, img_h = img.size

                    # Ширина и высота рамки YOLO в пикселях
                    box_w_px = w * img_w
                    box_h_px = h * img_h

                    # --- НАСТРОЙКА ЖЕСТКОГО КРОПА ---
                    # 0.20 означает, что мы отрезаем по 20% ширины слева и справа
                    # Увеличивайте эти цифры, если трубы все еще попадают в кадр!
                    # --- НЕЗАВИСИМАЯ НАСТРОЙКА КРОПА ДЛЯ КАЖДОЙ СТОРОНЫ ---
                    margin_left = 0.05  # 0.0 означает, что слева не отрезаем ничего
                    margin_right = 0.05  # 0.0 означает, что справа не отрезаем ничего
                    margin_top = 0.0  # 0.0 означает, что сверху не отрезаем ничего

                    margin_bottom = 0.35  # 🔥 Отрезаем 30% ТОЛЬКО СНИЗУ

                    left = int(
                        (x_c * img_w) - (box_w_px / 2) + (box_w_px * margin_left)
                    )
                    right = int(
                        (x_c * img_w) + (box_w_px / 2) - (box_w_px * margin_right)
                    )
                    top = int((y_c * img_h) - (box_h_px / 2) + (box_h_px * margin_top))
                    bottom = int(
                        (y_c * img_h) + (box_h_px / 2) - (box_h_px * margin_bottom)
                    )

                    # Защита от слишком сильного кропа (чтобы рамка не вывернулась наизнанку)
                    left = int(
                        (x_c * img_w) - (box_w_px / 2) + (box_w_px * margin_left)
                    )
                    right = int(
                        (x_c * img_w) + (box_w_px / 2) - (box_w_px * margin_right)
                    )
                    top = int((y_c * img_h) - (box_h_px / 2) + (box_h_px * margin_top))
                    bottom = int(
                        (y_c * img_h) + (box_h_px / 2) - (box_h_px * margin_bottom)
                    )

                    # Защита от выхода за границы
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(img_w, right), min(img_h, bottom)

                    if right > left and bottom > top:
                        img = img.crop((left, top, right, bottom))
                        # 🔥 ИСПРАВЛЕННАЯ СТРОКА 83: Убрали использование margin_x и margin_y
                        print("✂️ ЖЕСТКИЙ КРОП! Применены независимые отступы.")
                    else:
                        print("⚠️ Ошибка кропа: отступы слишком большие.")
    else:
        print("⚠️ ВНИМАНИЕ: Файл разметки не найден!")

    # Сохраняем вырезанный кусок, чтобы вы могли на него посмотреть
    img.save("cropped_input.jpg")

    # --- Стандартная обработка (ресайз и нормализация) ---
    img_resized = img.resize((size, size))
    img_viz = np.array(img_resized, dtype=np.float32) / 255.0

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor_img = transform(img)
    assert isinstance(tensor_img, torch.Tensor)
    input_tensor = tensor_img.unsqueeze(0)

    return img_viz, input_tensor, img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(
        name="convnext_tiny",
        pretrained=False,
    )
    model = CowBCSModel(cfg=model_cfg)

    weights_path = "/home/georgiy/projects/ml/cow/convnext_2xGPU_baseline purpul.pt"
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.removeprefix("module.")
        cleaned_state_dict[name] = v

    for k, v in list(cleaned_state_dict.items()):
        if v.ndim == 2 and v.shape[0] == 1:
            cleaned_state_dict["head.weight"] = v
        if v.ndim == 1 and v.shape[0] == 1 and ("bias" in k or "b" in k):
            cleaned_state_dict["head.bias"] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    backbone: Any = model.backbone
    target_layers = [backbone.stages[-1].blocks[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Укажите вашу картинку
    image_path = "/home/georgiy/projects/ml/cow/data/raw/images/train/0000_50_1_18-09-24-12-12-15-9.webp"

    img_viz, input_tensor, _ = load_image_tight_crop(image_path, size=384)
    input_tensor = input_tensor.to(device)

    targets = [RegressionTarget()]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # type: ignore
    grayscale_cam = grayscale_cam[0, :]

    with torch.no_grad():
        pred_bcs = model(input_tensor).item()
    print(f"🎯 Предсказанный BCS (на чистом кропе): {pred_bcs:.2f}")

    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    cv2.imwrite("cam_output_tight.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print("✅ Готово!")
    print("1. Посмотрите 'cropped_input.jpg' — чтобы убедиться, что трубы ушли.")
    print("2. Посмотрите 'cam_output_tight.jpg' — куда теперь смотрит модель.")


if __name__ == "__main__":
    main()
