import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import os

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from ultralytics import YOLO


# --- НОВЫЙ КЛАСС-ОБЕРТКА ---
class YOLOWrapper(nn.Module):
    """
    Обертка, которая прячет сложный выход YOLO от библиотеки Grad-CAM.
    Возвращает только первый тензор (основные предсказания).
    """

    def __init__(self, yolo_model: nn.Module):
        super().__init__()
        self.model = yolo_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # Если YOLO вернул кортеж или список, берем только первый элемент
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


# ---------------------------


def load_image_for_yolo(
    image_path: str, img_size: int = 640
) -> tuple[np.ndarray, torch.Tensor]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # YOLO работает в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- УМНЫЙ ПОИСК YOLO-ЛЕЙБЛА ДЛЯ КРОПА ---
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

                    # Размеры оригинальной картинки (высота, ширина)
                    img_h, img_w = img.shape[:2]

                    # Перевод из процентов в пиксели
                    left = int((x_c - w / 2) * img_w)
                    top = int((y_c - h / 2) * img_h)
                    right = int((x_c + w / 2) * img_w)
                    bottom = int((y_c + h / 2) * img_h)

                    # Защита от выхода за границы картинки
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(img_w, right), min(img_h, bottom)

                    # Вырезаем нужную область (в OpenCV это просто срез матрицы)
                    if right > left and bottom > top:
                        img = img[top:bottom, left:right]
                        print(
                            f"✂️ Успех! Корова вырезана по BBox из: {os.path.basename(label_path)}"
                        )
                    else:
                        print(
                            "⚠️ Ошибка координат BBox. Используется целое изображение."
                        )
    else:
        print("⚠️ ВНИМАНИЕ: Файл разметки не найден! Корова НЕ вырезана.")
        print(f"Искал пути:\n 1. {label_path_1}\n 2. {label_path_2}")

    # --- Стандартная обработка: ресайз и нормализация кропа ---
    img_resized = cv2.resize(img, (img_size, img_size))

    img_viz = np.array(img_resized, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(img_viz).permute(2, 0, 1).unsqueeze(0)

    return img_viz, input_tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = "/home/georgiy/projects/ml/cow/yolo26m_size_640_v1.pt"
    model = YOLO(weights_path)

    pytorch_model = model.model
    assert isinstance(pytorch_model, nn.Module)
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()

    # ОБОРАЧИВАЕМ МОДЕЛЬ В НАШ ВРАППЕР
    wrapped_model = YOLOWrapper(pytorch_model)

    layer_container = getattr(pytorch_model, "model", None)
    assert isinstance(layer_container, nn.Sequential)

    # Берем предпоследний слой из оригинальной YOLO
    target_layers = [layer_container[-2]]

    # Передаем в EigenCAM нашу ОБЕРНУТУЮ модель
    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)

    image_path = "data/raw/images/train/6883_115_1_17-05-24-13-46-39-0.webp"
    img_viz, input_tensor = load_image_for_yolo(image_path, img_size=640)
    input_tensor = input_tensor.to(device)

    grayscale_cam = cam(input_tensor=input_tensor)  # type: ignore
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    cv2.imwrite("yolo_cam_output.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("✅ Тепловая карта YOLO сохранена как yolo_cam_output.jpg")


if __name__ == "__main__":
    main()
