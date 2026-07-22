import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Импортируем нашу модель и словарь классов
from models import CowBCSModel
from data import DEFAULT_CLASS_TO_BCS


def predict_single_image(image_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # 1. Загружаем модель (с 17 классами для Ordinal Regression)
    model = CowBCSModel(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=False,
        num_classes=17,
    )

    # Загружаем веса (укажи правильный путь к твоему файлу .pth)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Если сохранял через DDP, ключи могут начинаться с "module." - очищаем их
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 2. Загружаем и готовим картинку
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить фото: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Применяем валидационные трансформации (размер должен совпадать с конфигом)
    transform = A.Compose(
        [
            A.Resize(height=384, width=384),  # Или 512x512, если ты менял в конфиге
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    input_tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    # 3. Делаем предсказание
    with torch.no_grad():
        logits = model(input_tensor)  # Получаем 16 логитов

        # Переводим логиты в вероятности (сигмоида)
        probs = torch.sigmoid(logits)

        # Считаем, сколько порогов преодолели барьер в 50%
        # Явное приведение к int для Pylance
        predicted_class_idx = int((probs > 0.5).sum(dim=1).item())

    # Переводим индекс обратно в балл BCS
    predicted_bcs = DEFAULT_CLASS_TO_BCS.get(predicted_class_idx, -1.0)

    # 4. Выводим результат на экран
    print(f"🔥 Предсказанный индекс класса: {predicted_class_idx}")
    print(f"🐄 Итоговая оценка BCS: {predicted_bcs}")

    # Показываем картинку
    plt.imshow(image)
    plt.title(f"Predicted BCS: {predicted_bcs}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Укажи путь к скачанной картинке и путь к лучшему чекпоинту
    TEST_IMAGE = "/home/georgiy/projects/ml/cow/tests/photo-of-body-condition-score-2-75-from-top62e610e251e9b.webp"
    MODEL_WEIGHTS = "/home/georgiy/projects/ml/cow/best_bcs_model.pt"

    predict_single_image(TEST_IMAGE, MODEL_WEIGHTS)
