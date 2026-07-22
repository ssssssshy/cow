from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

# --- 1. Маппинг классов в непрерывные значения BCS ---
DEFAULT_CLASS_TO_BCS: Dict[int, float] = {
    0: 1.0,
    1: 1.25,
    2: 1.5,
    3: 1.75,
    4: 2.0,
    5: 2.25,
    6: 2.5,
    7: 2.75,
    8: 3.0,
    9: 3.25,
    10: 3.5,
    11: 3.75,
    12: 4.0,
    13: 4.25,
    14: 4.5,
    15: 4.75,
    16: 5.0,
}


# --- 2. Конфигурация аугментаций (Albumentations) ---
def get_transforms(
    img_size: Tuple[int, int] = (384, 384),
) -> Tuple[A.Compose, A.Compose]:
    """Aугментации, оптимизированные под ракурс съемки коров сверху (Top-View).

    img_size: (Height, Width) для входа в нейросеть
    """
    train_transform = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.05, 0.05),
                rotate=(-12, 12),
                p=0.6,
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.GaussNoise(p=0.5),
                ],
                p=0.3,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


# --- 3. PyTorch Dataset ---
class CowBCSDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        img_size: Tuple[int, int] = (384, 384),
        crop_bbox: bool = True,
        bbox_padding: float = 0.1,
        transform: Optional[A.Compose] = None,
        class_to_bcs: Optional[Dict[int, float]] = None,
    ):
        """Dataset для задачи оценки упитанности коров (BCS)."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.crop_bbox = crop_bbox
        self.bbox_padding = bbox_padding
        self.transform = transform
        self.class_to_bcs = class_to_bcs or DEFAULT_CLASS_TO_BCS

        self.img_dir = self.data_dir / "images" / split
        self.lbl_dir = self.data_dir / "labels" / split

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []
        img_extensions = ("*.webp", "*.jpg", "*.jpeg", "*.png")
        img_files: List[Path] = []
        for ext in img_extensions:
            img_files.extend(list(self.img_dir.glob(ext)))

        for img_path in img_files:
            lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    continue

                parts = lines[0].split()
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                samples.append(
                    {
                        "img_path": img_path,
                        "class_id": cls_id,
                        "bcs_target": self.class_to_bcs.get(cls_id, float(cls_id)),
                        "bbox": (xc, yc, w, h),
                    }
                )

        print(f"[{self.split.upper()}] Успешно загружено образцов: {len(samples)}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        image = cv2.imread(str(sample["img_path"]))
        if image is None:
            raise FileNotFoundError(
                f"Не удалось прочитать картинку: {sample['img_path']}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image.shape

        if self.crop_bbox:
            xc, yc, bw, bh = sample["bbox"]

            pad_w = bw * self.bbox_padding
            pad_h = bh * self.bbox_padding

            xmin = max(0, int((xc - bw / 2 - pad_w) * w_img))
            ymin = max(0, int((yc - bh / 2 - pad_h) * h_img))
            xmax = min(w_img, int((xc + bw / 2 + pad_w) * w_img))
            ymax = min(h_img, int((yc + bh / 2 + pad_h) * h_img))

            image = image[ymin:ymax, xmin:xmax]

        if self.transform:
            augmented = self.transform(image=image)
            image_out = augmented["image"]
        else:
            image_out = image

        # Гарантируем для Pylance и PyTorch, что на выходе строго torch.Tensor
        if not isinstance(image_out, torch.Tensor):
            image_out = torch.from_numpy(image_out).permute(2, 0, 1).float() / 255.0

        bcs_target = torch.tensor(sample["bcs_target"], dtype=torch.float32)
        class_id = sample["class_id"]

        return image_out, bcs_target, class_id


# --- 4. Weighted Sampler для компенсации дисбаланса ---
def get_weighted_sampler(dataset: CowBCSDataset) -> WeightedRandomSampler:
    """Возвращает WeightedRandomSampler для балансировки батчей."""
    targets = [sample["bcs_target"] for sample in dataset.samples]
    counts = Counter(targets)

    sample_weights = [1.0 / counts[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# --- 5. Функция сборки DataLoaders ---
def get_dataloaders(
    data_dir: Union[str, Path] = "data/raw",
    batch_size: int = 16,
    img_size: Tuple[int, int] = (384, 384),
    crop_bbox: bool = True,
    num_workers: int = 4,
    is_distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:

    train_tf, val_tf = get_transforms(img_size=img_size)

    # ВАЖНО: Убедитесь, что датасет уже разделен по ID коровы, а не случайно[cite: 1].
    train_dataset = CowBCSDataset(
        data_dir,
        split="train",
        img_size=img_size,
        crop_bbox=crop_bbox,
        transform=train_tf,
    )
    val_dataset = CowBCSDataset(
        data_dir, split="val", img_size=img_size, crop_bbox=crop_bbox, transform=val_tf
    )

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# --- Quick Test ---
# --- Quick Test ---
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(
        data_dir="data/raw",
        batch_size=4,
        img_size=(384, 384),
        is_distributed=False,  # <-- Заменили параметр
    )

    images, targets, class_ids = next(iter(train_loader))
    print("\nТестовый батч:")
    print(f"  Формат картинок: {images.shape}")
    print(f"  Таргеты BCS: {targets}")
    print(f"  Class IDs: {class_ids}")
