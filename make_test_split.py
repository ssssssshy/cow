import random
import shutil
from pathlib import Path


def create_test_split(
    data_dir: str, num_test_samples: int = 10, source_split: str = "train"
):
    base_path = Path(data_dir)

    src_img_dir = base_path / "images" / source_split
    src_lbl_dir = base_path / "labels" / source_split

    test_img_dir = base_path / "images" / "test"
    test_lbl_dir = base_path / "labels" / "test"

    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = (".webp", ".jpg", ".jpeg", ".png")
    all_images = []
    for ext in img_extensions:
        all_images.extend(list(src_img_dir.glob(f"*{ext}")))

    if not all_images:
        print(f"Images not found in {src_img_dir}")
        return

    if len(all_images) < num_test_samples:
        print(
            f"Warning: requested {num_test_samples} samples, but only {len(all_images)} available. Taking all."
        )
        num_test_samples = len(all_images)

    selected_images = random.sample(all_images, num_test_samples)

    moved_count = 0
    for img_path in selected_images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"

        if lbl_path.exists():
            shutil.move(str(img_path), str(test_img_dir / img_path.name))

            shutil.move(str(lbl_path), str(test_lbl_dir / lbl_path.name))

            moved_count += 1
        else:
            print(f"Label not found for {img_path.name}, skipping.")

    print(
        f"Done. Moved {moved_count} image-label pairs from '{source_split}' to 'test'."
    )


if __name__ == "__main__":
    # Укажите нужные параметры
    create_test_split(
        data_dir="data/raw",
        num_test_samples=10,  # Сколько картинок отложить для теста
        source_split="train",  # Откуда забирать картинки (train или val)
    )
