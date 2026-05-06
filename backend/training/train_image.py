"""
Image Model Training Script
============================
Trains EfficientNet-B4 + SRM + Frequency Head deepfake detector.

Usage:
    python -m backend.training.train_image \
        --train_dir data/train \
        --val_dir   data/val \
        --epochs    30 \
        --batch_size 32 \
        --lr        2e-4
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from backend.models.image_detector import DinoV2DeepfakeDetector, ImageDeepfakeDetector, LightImageDetector
from backend.training.dataset import build_image_loaders
from backend.training.losses import CombinedLoss
from backend.training.config import TrainConfig, DataConfig


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay   = decay
        self.shadow  = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# One Epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_labels, all_scores = [], []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type=device, enabled=(device == "cuda")):
                out  = model(images)
                loss = criterion(out["logit"].squeeze(1), labels)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(out["score"].squeeze(1).cpu().detach().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
    preds    = [1 if s > 0.5 else 0 for s in all_scores]
    f1       = f1_score(all_labels, preds, zero_division=0)

    return {"loss": avg_loss, "auc": auc, "f1": f1}


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(
    train_dir:  str,
    val_dir:    str,
    save_dir:   str = "checkpoints",
    epochs:     int = 30,
    batch_size: int = 32,
    lr:         float = 2e-4,
    device:     str = "auto",
    seed:       int = 42,
    image_size: int = 224,
    num_workers:int = 4,
    model_arch: str = "light",
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    set_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, _ = build_image_loaders(
        train_dir=train_dir, val_dir=val_dir, test_dir=val_dir,
        batch_size=batch_size, image_size=image_size, num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────
    if model_arch == "full":
        model = ImageDeepfakeDetector(pretrained=pretrained).to(device)
    elif model_arch == "light":
        model = LightImageDetector(pretrained=pretrained).to(device)
    elif model_arch == "dinov2-large":
        model = DinoV2DeepfakeDetector(pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    else:
        raise ValueError(f"Unknown model_arch: {model_arch}")
    if model_arch == "light":
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size, device=device))
    ema   = EMA(model, decay=0.999)

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = GradScaler(device=device, enabled=(device == "cuda"))
    criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    if HAS_MLFLOW:
        mlflow.start_run()
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "image_size": image_size,
            "num_workers": num_workers,
            "model_arch": model_arch,
            "freeze_backbone": freeze_backbone,
        })

    print(f"\n{'='*60}")
    print(f"Training {model_arch} Image Deepfake Detector for {epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, is_train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, is_train=False)
        ema.update(model)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"AUC {train_metrics['auc']:.4f}/{val_metrics['auc']:.4f} | "
            f"F1 {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if HAS_MLFLOW:
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
                "train_auc":  train_metrics["auc"],  "val_auc":  val_metrics["auc"],
            }, step=epoch)

        # Save best checkpoint
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt_path = Path(save_dir) / f"image_{model_arch}_best.pt"
            torch.save({
                "epoch":       epoch,
                "model_arch":  model_arch,
                "image_size":  image_size,
                "hf_model":    "facebook/dinov2-large" if model_arch == "dinov2-large" else None,
                "model_state": model.state_dict(),
                "ema_shadow":  ema.shadow,
                "val_auc":     best_auc,
                "optimizer":   optimizer.state_dict(),
            }, ckpt_path)
            print(f"  [OK] Saved best model (AUC={best_auc:.4f}) -> {ckpt_path}")

    if HAS_MLFLOW:
        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.end_run()

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    return best_auc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image deepfake detector")
    parser.add_argument("--train_dir",  required=True)
    parser.add_argument("--val_dir",    required=True)
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--image_size", type=int,   default=224)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--model_arch", choices=["light", "full", "dinov2-large"], default="light")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Train only the classifier head; recommended for quick DINOv2 CPU tests")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir, val_dir=args.val_dir,
        save_dir=args.save_dir,   epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        device=args.device,       seed=args.seed,
        image_size=args.image_size, num_workers=args.num_workers,
        model_arch=args.model_arch, pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )
