import shutil
from pathlib import Path

import pandas as pd


def collect_hard_images(
    csv_path="/home/georgiy/projects/ml/cow/data/processed/hard_examples_report.csv",
    output_dir="results/review_queue",
    top_n=50,
):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(
            f"❌ Файл {csv_path} не найден! Сначала запусти python src/eval.py локально."
        )
        return

    df = pd.read_csv(csv_file)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    # Твой локальный базовый путь к датасету
    base_dir = Path("/home/georgiy/projects/ml/cow").resolve()
    images_base = base_dir / "data" / "raw" / "images"

    print(
        f"📂 Обрабатываем пути и копируем топ-{min(top_n, len(df))} проблемных фото..."
    )

    count = 0
    for idx, row in df.head(top_n).iterrows():
        old_path = Path(row["image_path"])
        filename = old_path.name

        # Определяем сплит (val или train) из старого пути
        split = "val" if "val" in old_path.parts else "train"

        # Собираем правильный локальный путь
        local_img_path = images_base / split / filename

        if not local_img_path.exists():
            print(f"⚠️ Не найдено локально: {local_img_path}")
            continue

        # Обновляем путь в датафрейме на локальный
        df.loc[idx, "image_path"] = str(local_img_path)

        # Достаем метрики для красивого имени файла при копировании
        error = row["error"]
        true_bcs = row["true_bcs"]
        pred_bcs = row["pred_bcs"]

        new_filename = (
            f"err_{error:.2f}_true_{true_bcs:.2f}_pred_{pred_bcs:.2f}_{filename}"
        )
        dest_path = out_path / new_filename

        shutil.copy(local_img_path, dest_path)
        count += 1

    # Сохраняем исправленный CSV обратно
    df.to_csv(csv_file, index=False)

    print(f"\n✅ Успешно скопировано файлов: {count}")
    print(f"📁 Папка для проверки: {out_path.absolute()}")
    print("💡 Открой эту папку в файловом менеджере и оцени сложные примеры!")


if __name__ == "__main__":
    collect_hard_images()
