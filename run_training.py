"""
Real Data Training Pipeline — CPU-Optimized
=============================================
Downloads publicly available deepfake detection datasets and runs training.

DATASETS (no registration required):
  1. CIFAKE (HuggingFace) — 60,000 real CIFAR-10 vs AI-generated images
  2. 140k Real/Fake Faces (optional, Kaggle API)
  3. FaceForensics++ frames (optional, requires form registration)

Usage:
    python run_training.py --dataset cifake --epochs 20 --batch_size 16
    python run_training.py --dataset custom --data_dir data/ --epochs 30
"""

import os
import sys
import argparse
import time
from pathlib import Path

# ── Force CPU if no CUDA ──────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Training device: {DEVICE.upper()}")
if DEVICE == "cpu":
    # Maximize CPU parallelism
    torch.set_num_threads(min(os.cpu_count(), 12))
    print(f"[*] CPU threads: {torch.get_num_threads()}")


def main():
    parser = argparse.ArgumentParser(description="DeepShield Training Pipeline")
    parser.add_argument("--dataset",    choices=["cifake", "ff++", "custom"], default="cifake")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--model",      choices=["image", "video", "audio", "all"], default="image")
    parser.add_argument("--model_arch", choices=["light", "full", "dinov2-large"], default="light",
                        help="light is fastest on CPU; full uses EfficientNet-B4; dinov2-large uses facebook/dinov2-large")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Train only the classifier head; useful for quick DINOv2 experiments")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Do not download pretrained backbone weights")
    parser.add_argument("--image_size", type=int,   default=128,
                        help="Use 128 for CPU training (faster), 224 for full model")
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--max_samples",type=int,   default=None,
                        help="Limit dataset size for quick experiments")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if data already exists")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  DeepShield Training Pipeline")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Model:      {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}")
    print(f"  Device:     {DEVICE}")
    print(f"{'='*60}\n")

    # ── Step 1: Download dataset ──────────────────────────────────
    if not args.skip_download:
        if args.dataset == "cifake":
            from data_pipeline.download_cifake import download_cifake
            download_cifake(args.data_dir, max_samples=args.max_samples)

        elif args.dataset == "ff++":
            from data_pipeline.download_ffpp import download_ffpp_frames
            download_ffpp_frames(args.data_dir)

        elif args.dataset == "custom":
            print(f"[*] Using custom data from: {args.data_dir}")
            print(f"    Expected structure: {args.data_dir}/train/real/, {args.data_dir}/train/fake/")
    else:
        print("[*] Skipping download (--skip_download set)")

    # ── Step 2: Train ─────────────────────────────────────────────
    if args.model in ("image", "all"):
        print("\n[*] Training Image Deepfake Detector...")
        from backend.training.train_image import train as train_image
        train_image(
            train_dir  = str(Path(args.data_dir) / "train"),
            val_dir    = str(Path(args.data_dir) / "val"),
            save_dir   = args.save_dir,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            device     = DEVICE,
            image_size = args.image_size,
            num_workers= args.num_workers,
            model_arch = args.model_arch,
            pretrained = not args.no_pretrained,
            freeze_backbone = args.freeze_backbone,
        )

    if args.model in ("video", "all"):
        print("\n[*] Training Video Deepfake Detector...")
        from backend.training.train_video import train as train_video
        train_video(
            data_dir   = args.data_dir,
            save_dir   = args.save_dir,
            epochs     = args.epochs,
            batch_size = max(1, args.batch_size // 4),  # videos are heavier
            device     = DEVICE,
        )

    print("\n[OK] Training complete!")
    print(f"[*] Checkpoints saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
