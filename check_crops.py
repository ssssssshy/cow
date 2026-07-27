import cv2
import numpy as np
import matplotlib.pyplot as plt

# Укажите имена ваших файлов здесь
IMG_PATH = "/home/georgiy/projects/ml/cow/0000_50_1_18-09-24-12-13-44-2.webp"  # или .jpg / .png
TXT_PATH = "/home/georgiy/projects/ml/cow/0000_116_1_03-06-24-12-42-56-9.txt"

# Загружаем картинку
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"❌ Не удалось загрузить картинку: {IMG_PATH}")
    exit()

# OpenCV читает в BGR, переводим в RGB для правильного отображения цветов
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img.shape

# Создаем пустую (черную) маску размером с картинку
mask = np.zeros((H, W), dtype=np.uint8)

# Читаем файл разметки
try:
    with open(TXT_PATH, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"❌ Не найден файл разметки: {TXT_PATH}")
    exit()

for line in lines:
    parts = line.strip().split()
    # Если чисел много — это полигон сегментации
    if len(parts) > 5:
        # Пропускаем первый элемент (ID класса) и берем только координаты
        coords = [float(p) for p in parts[1:]]

        # Переводим из относительных процентов (0..1) в реальные пиксели
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * W)
            y = int(coords[i + 1] * H)
            points.append([x, y])

        pts = np.array(points, dtype=np.int32)

        # Рисуем белый полигон на нашей черной маске
        cv2.fillPoly(mask, [pts], 255)

# Применяем маску к картинке (оставляем только то, что внутри полигона)
masked_cow = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# Делаем плотный кроп (обрезаем лишний черный фон по краям)
y_indices, x_indices = np.where(mask > 0)
if len(y_indices) > 0 and len(x_indices) > 0:
    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    cropped_cow = masked_cow[y1:y2, x1:x2]
else:
    cropped_cow = masked_cow

# --- Отрисовка на экране ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("1. Оригинал")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("2. Вырезанный фон")
plt.imshow(masked_cow)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("3. Финальный кроп")
plt.imshow(cropped_cow)
plt.axis("off")

plt.tight_layout()
# Откроет окно на вашем ПК с результатом
plt.show()
