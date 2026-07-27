import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.config import ModelConfig
from src.models import CowBCSModel


# 🔥 Тот самый фикс: просто возвращаем скаляр, который выдала модель
class RegressionTarget:
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output


def load_image(image_path: str, size: int = 384) -> tuple[np.ndarray, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")

    # --- УМНЫЙ ПОИСК YOLO-ЛЕЙБЛА ---
    # Вариант 1: файл .txt лежит прямо рядом с картинкой
    label_path_1 = os.path.splitext(image_path)[0] + ".txt"
    # Вариант 2: стандартная структура YOLO (заменяем папку images на labels)
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
                # Берем первую строчку (вдруг там несколько рамок)
                line = lines[0].strip().split()
                if len(line) >= 5:
                    _, x_c, y_c, w, h = map(float, line[:5])

                    img_w, img_h = img.size

                    # YOLO: переводим проценты в абсолютные пиксели
                    left = int((x_c - w / 2) * img_w)
                    top = int((y_c - h / 2) * img_h)
                    right = int((x_c + w / 2) * img_w)
                    bottom = int((y_c + h / 2) * img_h)

                    # Защита от выхода за границы
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(img_w, right), min(img_h, bottom)

                    # Кропаем!
                    img = img.crop((left, top, right, bottom))
                    print(
                        f"✂️ Успех! Корова вырезана по BBox из: {os.path.basename(label_path)}"
                    )
                    print(f"📐 Новый размер картинки до ресайза: {img.size}")
    else:
        print("⚠️ ВНИМАНИЕ: Файл разметки не найден! Корова НЕ вырезана.")
        print(f"Искал пути:\n 1. {label_path_1}\n 2. {label_path_2}")

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

    return img_viz, input_tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 Исправлено: передаем через ModelConfig
    model_cfg = ModelConfig(
        name="convnext_tiny",  # Укажите то же имя, на котором обучали
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

    # Прикручиваем веса от старого бейзлайна к новой архитектуре
    for k, v in list(cleaned_state_dict.items()):
        if v.ndim == 2 and v.shape[0] == 1:
            cleaned_state_dict["head.weight"] = v
        if v.ndim == 1 and v.shape[0] == 1 and ("bias" in k or "b" in k):
            cleaned_state_dict["head.bias"] = v

    # Загружаем (strict=False проигнорирует лишнее)
    model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Снимаем градиенты с бэкбона (актуально для ConvNeXt)
    target_layers = [model.backbone.stages[-1].blocks[-1]]  # type: ignore

    cam = GradCAM(model=model, target_layers=target_layers)

    image_path = "/home/georgiy/projects/ml/cow/data/raw/images/train/0000_50_1_18-09-24-12-12-15-9.webp"
    img_viz, input_tensor = load_image(image_path, size=384)
    input_tensor = input_tensor.to(device)

    # Используем нашу кастомную функцию
    targets = [RegressionTarget()]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # type: ignore
    grayscale_cam = grayscale_cam[0, :]

    with torch.no_grad():
        pred_bcs = model(input_tensor).item()
    print(f"🎯 Предсказанный BCS: {pred_bcs:.2f}")

    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    cv2.imwrite("cam_output.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("📸 Тепловая карта сохранена как cam_output.jpg")


if __name__ == "__main__":
    main()
