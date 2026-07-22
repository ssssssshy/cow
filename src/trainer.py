import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from data import get_dataloaders
from losses import get_loss_function
from metrics import compute_all_metrics, compute_mae
from models import CowBCSModel
from torch.utils.data.distributed import DistributedSampler
from utils import EarlyStopping


def setup_ddp():
    """Инициализация распределенного обучения."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    # ... (остальной код функции без изменений)

    running_loss, running_mae, total_samples = 0.0, 0.0, 0

    # Показывать прогресс-бар только на нулевом GPU (Master Node)
    is_master = int(os.environ.get("LOCAL_RANK", 0)) == 0
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False) if is_master else loader

    for images, targets, _ in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP)
        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Защита от грязных меток с помощью Gradient Clipping max_norm=10.0[cite: 1]
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        batch_mae = compute_mae(preds, targets)
        running_loss += loss.item() * batch_size
        running_mae += batch_mae * batch_size
        total_samples += batch_size

        if is_master:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mae": f"{batch_mae:.3f}"})

    return running_loss / total_samples, running_mae / total_samples


@torch.inference_mode()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []

    is_master = int(os.environ.get("LOCAL_RANK", 0)) == 0
    pbar = tqdm(loader, desc="Val  ", leave=False) if is_master else loader

    for images, targets, _ in pbar:
        images, targets = images.to(device), targets.to(device)

        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)
            loss = criterion(preds, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(preds)
        all_targets.append(targets)

    # В DDP нужно собрать предсказания со всех GPU
    concat_preds = torch.cat(all_preds, dim=0)
    concat_targets = torch.cat(all_targets, dim=0)

    gathered_preds = [
        torch.zeros_like(concat_preds) for _ in range(dist.get_world_size())
    ]
    gathered_targets = [
        torch.zeros_like(concat_targets) for _ in range(dist.get_world_size())
    ]

    dist.all_gather(gathered_preds, concat_preds)
    dist.all_gather(gathered_targets, concat_targets)

    final_preds = torch.cat(gathered_preds, dim=0)
    final_targets = torch.cat(gathered_targets, dim=0)

    val_loss = running_loss / total_samples

    # Метрики считаем только на Master Node
    val_metrics = compute_all_metrics(final_preds, final_targets) if is_master else None

    return val_loss, val_metrics


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = local_rank == 0

    # Конфигурация
    DATA_DIR = Path("data/raw")
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k_384"
    LOSS_NAME = "smooth_l1"
    IMG_SIZE = (384, 384)
    BATCH_SIZE = 32  # Batch per GPU
    EPOCHS = 50
    LR = 1e-4
    WARMUP_EPOCHS = 3  # Разогрев[cite: 1]
    PATIENCE = 7

    SAVE_DIR = Path("checkpoints")
    if is_master:
        SAVE_DIR.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        is_distributed=True,
        num_workers=4,
    )

    model = CowBCSModel(model_name=MODEL_NAME, pretrained=True).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Используем Cost-Sensitive Learning вместо Weighted Sampler[cite: 1]
    criterion = get_loss_function(LOSS_NAME).to(device)

    # Оптимизатор AdamW с разогревом и косинусным затуханием[cite: 1]
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    scaler = torch.amp.GradScaler("cuda")
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_val_mae = float("inf")

    if is_master:
        print("\n" + "=" * 70)
        print(f"🎯 DDP Обучение | GPUs: {dist.get_world_size()} | Модель: {MODEL_NAME}")
        print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()
        early_stopping(val_loss)

        if is_master and val_metrics is not None:
            elapsed = time.time() - start_time
            val_mae = val_metrics["mae"]
            acc_025 = val_metrics["acc_tol_0.25"]

            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae
                torch.save(model.module.state_dict(), SAVE_DIR / "best_bcs_model.pt")

            print(
                f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} MAE: {train_mae:.3f} | "
                f"Val MAE: {val_mae:.3f} Acc(±0.25): {acc_025:.1f}% {'🔥 BEST' if is_best else ''}"
            )

        if early_stopping.early_stop:
            if is_master:
                print("🛑 Сработал Early Stopping! Обучение остановлено.")
            break

    cleanup_ddp()


if __name__ == "__main__":
    main()
