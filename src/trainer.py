import os
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

from src.config import Config
from src.data import get_dataloaders
from src.losses import get_loss_function
from src.metrics import compute_all_metrics, compute_mae
from src.models import CowBCSModel
from src.utils import EarlyStopping


def setup_ddp():
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

    running_loss, running_mae, total_samples = 0.0, 0.0, 0

    is_master = int(os.environ.get("LOCAL_RANK", 0)) == 0
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False) if is_master else loader

    for images, targets, _ in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            preds = model(images)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
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


@torch.no_grad()
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

    # 🔥 Синхронизируем running_loss и total_samples между всеми GPU для честного Val Loss
    stats_tensor = torch.tensor([running_loss, float(total_samples)], device=device)
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    global_val_loss = stats_tensor[0].item() / stats_tensor[1].item()

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

    val_metrics = compute_all_metrics(final_preds, final_targets)

    # Расчет MAE только для экстремальных классов
    extreme_mask = (final_targets <= 2.50) | (final_targets >= 3.50)
    if extreme_mask.any():
        val_metrics["extreme_mae"] = compute_mae(
            final_preds[extreme_mask], final_targets[extreme_mask]
        )
    else:
        val_metrics["extreme_mae"] = 0.0

    return global_val_loss, val_metrics


def run_training(cfg: Config):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = local_rank == 0

    SAVE_DIR = Path(cfg.train.save_dir)
    if is_master:
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        if cfg.train.use_wandb:
            wandb.init(
                project=cfg.train.wandb_project,
                name=cfg.train.wandb_name,
                config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            )

    train_loader, val_loader = get_dataloaders(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.train.batch_size,
        img_size=(
            cfg.data.img_size[0],
            cfg.data.img_size[1],
        ),
        crop_bbox=cfg.data.crop_bbox,
        is_distributed=True,
        num_workers=cfg.data.num_workers,
        mixup_alpha=cfg.train.mixup_alpha,
        target_noise=cfg.data.target_noise,
    )

    model = CowBCSModel(
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        drop_rate=cfg.model.drop_rate,
        init_bias=cfg.model.init_bias,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = get_loss_function(cfg.train.loss_name).to(device)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Отключаем WD для одномерных тензоров (LayerNorm, LayerScale) и всех смещений (bias)
        if param.ndim < 2 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            # Применяем WD только к матрицам весов (Conv2d, Linear)
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": cfg.train.weight_decay},
        {
            "params": no_decay_params,
            "weight_decay": 0.0,  # Полностью отключаем штраф
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.train.lr)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.train.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs - cfg.train.warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.train.warmup_epochs],
    )

    scaler = torch.amp.GradScaler("cuda")
    early_stopping = EarlyStopping(patience=cfg.train.patience)

    # 🔥 Инициализация SWA
    if cfg.train.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.train.swa_lr)

    best_val_mae = float("inf")

    if is_master:
        print("\n" + "=" * 70)
        print(
            f"🎯 DDP Обучение | GPUs: {dist.get_world_size()} | Модель: {cfg.model.name}"
        )
        print("=" * 70)

    for epoch in range(1, cfg.train.epochs + 1):
        start_time = time.time()

        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Вытаскиваем MAE для Early Stopping
        val_mae = val_metrics["mae"]

        # Шаг планировщика (обычный или SWA)
        if cfg.train.use_swa and epoch >= cfg.train.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # 🔥 ТЕПЕРЬ СЛЕДИМ ЗА MAE, А НЕ ЗА LOSS
        early_stopping(val_mae)

        if is_master:
            elapsed = time.time() - start_time
            extreme_mae = val_metrics.get("extreme_mae", 0.0)
            acc_025 = val_metrics["acc_tol_0.25"]

            if cfg.train.use_wandb:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/mae": train_mae,
                        "val/loss": val_loss,
                        "val/mae": val_mae,
                        "val/extreme_mae": extreme_mae,
                        "val/acc_0.25": acc_025,
                        "val/acc_0.50": val_metrics["acc_tol_0.50"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae
                torch.save(model.module.state_dict(), SAVE_DIR / "best_bcs_model.pt")
                if cfg.train.use_wandb:
                    wandb.save(str(SAVE_DIR / "best_bcs_model.pt"))

            print(
                f"Epoch [{epoch:02d}/{cfg.train.epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} MAE: {train_mae:.3f} | "
                f"Val MAE: {val_mae:.3f} (Extr: {extreme_mae:.3f}) Acc(±0.25): {acc_025:.1f}% {'🔥 BEST' if is_best else ''}"
            )

        if early_stopping.early_stop:
            if is_master:
                print("🛑 Сработал Early Stopping! Обучение остановлено.")
            break

    # 🔥 Финализация SWA (обновление BatchNorm)
    if cfg.train.use_swa:
        if is_master:
            print("🔄 Финализация SWA: Обновление статистики BatchNorm...")
        update_bn(train_loader, swa_model, device=device)

        # Валидация SWA модели
        val_loss_swa, val_metrics_swa = validate_epoch(
            swa_model, val_loader, criterion, device
        )
        if is_master and val_metrics_swa:
            print(
                f"📊 SWA Metrics | MAE: {val_metrics_swa['mae']:.3f} (Extr: {val_metrics_swa['extreme_mae']:.3f})"
            )
            torch.save(swa_model.module.state_dict(), SAVE_DIR / "swa_bcs_model.pt")

    if is_master and cfg.train.use_wandb:
        wandb.finish()

    cleanup_ddp()
