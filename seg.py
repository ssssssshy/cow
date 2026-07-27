from pathlib import Path
import cv2

# Указываем раздельные пути, опираясь на вашу структуру проекта
IMAGES_ROOT = Path("/home/georgiy/projects/ml/cow/data/raw/images")
LABELS_ROOT = Path("/home/georgiy/projects/ml/cow/data/raw/labels")
OUTPUT_CROPS_ROOT = Path(
    "/home/georgiy/projects/ml/cow/data/raw/crops"
)  # 🔥 Отдельная папка для кропов!

for split in ["train", "val"]:
    # По вашей структуре train/val лежат прямо внутри images и labels
    img_dir = IMAGES_ROOT / split
    lbl_dir = LABELS_ROOT / split
    out_crop_dir = OUTPUT_CROPS_ROOT / split

    out_crop_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"⚠️ Папки для сплита '{split}' не найдены, пропускаем.")
        continue

    processed_count = 0

    for img_path in img_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        txt_path = lbl_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue  # Если для картинки нет файла разметки — пропускаем

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W, _ = img.shape

        with open(txt_path, "r") as f:
            lines = f.readlines()

        cow_idx = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Читаем данные из YOLO формата
            cls_id = int(parts[0])  # noqa: F841
            x_center = float(parts[1])
            y_center = float(parts[2])
            w_box = float(parts[3])
            h_box = float(parts[4])

            # Переводим относительные координаты YOLO в абсолютные пиксели
            x1 = int((x_center - w_box / 2) * W)
            y1 = int((y_center - h_box / 2) * H)
            x2 = int((x_center + w_box / 2) * W)
            y2 = int((y_center + h_box / 2) * H)

            # Защита от выхода за границы изображения
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            # Вырезаем кроп
            cropped_cow = img[y1:y2, x1:x2]

            if cropped_cow.size > 0:
                crop_name = f"{img_path.stem}_cow_{cow_idx}.jpg"
                cv2.imwrite(str(out_crop_dir / crop_name), cropped_cow)
                cow_idx += 1
                processed_count += 1

    print(f"✅ Сплит '{split}': успешно вырезано кропов коров: {processed_count}")

print("\n🎉 Нарезка кропов по вашим разметкам завершена!")
