from collections import Counter
from math import ceil
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

DEFAULT_CLASS_TO_BCS: dict[int, float] = {
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


def get_transforms(
    img_size: tuple[int, int] = (384, 384),
) -> tuple[A.Compose, A.Compose]:

    train_transform = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.06, 0.06),
                rotate=(-15, 15),
                p=0.7,
            ),
            A.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08, p=0.6
            ),
            A.PixelDropout(dropout_prob=0.05, per_channel=True, p=0.5),
            A.RandomGamma(gamma_limit=(75, 125), p=0.4),
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


class CowBCSDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        img_size: tuple[int, int] = (384, 384),
        crop_bbox: bool = True,
        bbox_padding: float = 0.1,
        transform: A.Compose | None = None,
        class_to_bcs: dict[int, float] | None = None,
        target_noise: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.crop_bbox = crop_bbox
        self.bbox_padding = bbox_padding
        self.transform = transform
        self.class_to_bcs = class_to_bcs or DEFAULT_CLASS_TO_BCS
        self.target_noise = target_noise

        self.img_dir = self.data_dir / "images" / split
        self.lbl_dir = self.data_dir / "labels" / split
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        # ... (rest of the method unchanged) ...
        samples = []
        img_extensions = ("*.webp", "*.jpg", "*.jpeg", "*.png")
        img_files: list[Path] = []
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
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

        if not isinstance(image_out, torch.Tensor):
            image_out = torch.from_numpy(image_out).permute(2, 0, 1).float() / 255.0

        bcs_target_val = sample["bcs_target"]

        # Добавляем Гауссов шум к таргету для защиты от шума разметки (только при обучении)
        if self.split == "train" and self.target_noise > 0:
            import numpy as np

            bcs_target_val += np.random.normal(0, self.target_noise)
            # Ограничиваем таргет физическими рамками шкалы BCS
            bcs_target_val = np.clip(bcs_target_val, 1.0, 5.0)

        bcs_target = torch.tensor(bcs_target_val, dtype=torch.float32)
        class_id = sample["class_id"]
        return image_out, bcs_target, class_id


# --- КАСТОМНЫЙ СЕМПЛЕР: DDP + ВЕСА КЛАССОВ ---
class DistributedWeightedRandomSampler(torch.utils.data.Sampler):
    """Семплер, объединяющий WeightedRandomSampler и распределенное обучение (DDP)."""

    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement

        # Считаем веса для каждого элемента датасета
        class_ids = [sample["class_id"] for sample in dataset.samples]
        counts = Counter(class_ids)
        weights = [1.0 / (counts[cid] ** 0.7) for cid in class_ids]

        # Обратная частота для балансировки (смягченная с помощью counts ** 0.7)
        weights = [1.0 / (counts[t] ** 0.7) for t in class_ids]
        self.weights = torch.as_tensor(weights, dtype=torch.double)

        self.num_samples = len(class_ids)
        # Округляем количество сэмплов на реплику, чтобы всем хватило поровну
        self.total_size = ceil(self.num_samples / self.num_replicas) * self.num_replicas
        self.num_samples_per_replica = self.total_size // self.num_replicas
        self.epoch = 0

    def __iter__(self):
        # Синхронизируем генератор по эпохам, чтобы на каждой эпохе выборка менялась корректно
        generator = torch.Generator()
        generator.manual_seed(self.epoch)

        # Генерируем взвешенные индексы глобально
        indices = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=generator
        ).tolist()

        # Разделяем индексы между видеокартами (каждому GPU достается свой срез)
        indices = indices[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# --- COLLATE ФУНКЦИЯ ДЛЯ ВНЕДРЕНИЯ MIXUP ---
class MixupCollate:
    """Интерполяция изображений и таргетов (Mixup) на уровне формирования батча."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
        images, bcs_targets, class_ids = zip(*batch)

        images = torch.stack(images, dim=0)
        bcs_targets = torch.stack(bcs_targets, dim=0)
        class_ids = torch.tensor(class_ids, dtype=torch.long)

        if self.alpha > 0:
            # Сэмплируем лямбду из бета-распределения
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0

        batch_size = images.size(0)
        # Генерируем случайные индексы для перемешивания батча
        index = torch.randperm(batch_size)

        # Смешиваем изображения
        mixed_images = lam * images + (1 - lam) * images[index]
        # Смешиваем таргеты (в случае регрессии мы можем смешивать сами значения BCS)
        mixed_targets = lam * bcs_targets + (1 - lam) * bcs_targets[index]

        # Возвращаем смешанные изображения и таргеты.
        # Исходные class_ids возвращаем без изменений (они нужны в основном для метрик).
        return mixed_images, mixed_targets, class_ids


# --- Функция сборки DataLoaders ---
def get_dataloaders(
    data_dir,
    batch_size,
    img_size,
    crop_bbox=True,
    is_distributed=False,
    num_workers=4,
    mixup_alpha=0.2,  # Параметр для контроля интенсивности Mixup
    target_noise=0.0,
):
    train_tf, val_tf = get_transforms(img_size=img_size)

    train_dataset = CowBCSDataset(
        data_dir=data_dir,
        split="train",
        img_size=img_size,
        crop_bbox=crop_bbox,
        transform=train_tf,
        target_noise=target_noise,
    )
    val_dataset = CowBCSDataset(
        data_dir=data_dir,
        split="val",
        img_size=img_size,
        crop_bbox=crop_bbox,
        transform=val_tf,
    )

    # Инициализируем Collate функцию для Mixup (применяем только к обучающей выборке)
    train_collate_fn = MixupCollate(alpha=mixup_alpha) if mixup_alpha > 0 else None

    if is_distributed:
        # Используем наш гибридный семплер для DDP + балансировки классов
        train_sampler = DistributedWeightedRandomSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Для одиночной видеокарты используем стандартный WeightedRandomSampler
        targets = [sample["bcs_target"] for sample in train_dataset.samples]
        counts = Counter(targets)
        # Смягчение весов (counts ** 0.7)
        sample_weights = [1.0 / (counts[t] ** 0.7) for t in targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
