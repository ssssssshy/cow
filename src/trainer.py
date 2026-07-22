from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_dataloaders
from losses import get_loss_function
from metrics import compute_all_metrics, compute_mae
from models import CowBCSModel


# --- 1. Функция эпохи обучения ---
def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixed Precision forward pass
        with torch.amp.autocast(
            device_type=device.type, enabled=(device.type == "cuda")
        ):
            preds = model(images)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()

        # Unscale для корректного клиппинга градиентов
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        batch_mae = compute_mae(preds, targets)

        running_loss += loss.item() * batch_size
        running_mae += batch_mae * batch_size
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mae": f"{batch_mae:.3f}"})

    return running_loss / total_samples, running_mae / total_samples


# --- 2. Функция эпохи валидации ---
@torch.inference_mode()
def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)

        with torch.amp.autocast(
            device_type=device.type, enabled=(device.type == "cuda")
        ):
            preds = model(images)
            loss = criterion(preds, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(preds.detach())
        all_targets.append(targets.detach())

    val_loss = running_loss / total_samples

    # Полный расчет метрик по всем предсказаниям валидации
    concat_preds = torch.cat(all_preds, dim=0)
    concat_targets = torch.cat(all_targets, dim=0)
    val_metrics = compute_all_metrics(concat_preds, concat_targets)

    return val_loss, val_metrics


# --- 3. Точка входа ---
def main():
    # --- Конфигурация ---
    DATA_DIR = Path("data/raw")
    MODEL_NAME = "convnext_small.fb_in22k_ft_in1k_384"
    LOSS_NAME = "smooth_l1"  # Доступно: 'smooth_l1', 'wing', 'l1', 'mse'
    IMG_SIZE = (384, 384)
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    SAVE_DIR = Path("checkpoints")
    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Запуск обучения на устройстве: {device}")

    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        crop_bbox=True,
        use_weighted_sampler=True,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    # Model, Loss, Optimizer, Scheduler, Scaler
    model = CowBCSModel(
        model_name=MODEL_NAME,
        pretrained=True,
        drop_rate=0.2,
        init_bias=2.8,
    ).to(device)

    criterion = get_loss_function(LOSS_NAME)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")

    print("\n" + "=" * 70)
    print(f"🎯 Модель: {MODEL_NAME} | Loss: {LOSS_NAME} | Эпох: {EPOCHS}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start_time

        val_mae = val_metrics["mae"]
        acc_025 = val_metrics["acc_tol_0.25"]

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "model_name": MODEL_NAME,
                },
                SAVE_DIR / "best_bcs_model.pt",
            )

        best_mark = "🔥 BEST" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} MAE: {train_mae:.3f} | "
            f"Val MAE: {val_mae:.3f} Acc(±0.25): {acc_025:.1f}% {best_mark}"
        )

    print("\n" + "=" * 70)
    print(f"🎉 Обучение завершено! Лучший Val MAE: {best_val_mae:.3f}")
    print(f"💾 Модель сохранена в: {SAVE_DIR / 'best_bcs_model.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
