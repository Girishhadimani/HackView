"""
CIFAKE Dataset Downloader
==========================
Downloads CIFAKE from HuggingFace Hub — no account required.

CIFAKE: Real and AI-Generated Synthetic Images
  Real:  CIFAR-10 (50,000 train + 10,000 test) — 32×32 real photos
  Fake:  Stable Diffusion v1.4 generated equivalents (same splits)

Paper: https://arxiv.org/abs/2303.14126
HuggingFace: https://huggingface.co/datasets/cifake-real-and-ai-generated

IMPORTANT: CIFAKE images are 32×32. We upscale to the training size.
For a more challenging face-specific dataset, see download_140k.py
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import numpy as np


def create_demo_dataset(output_dir: str, max_samples: Optional[int] = None) -> str:
    """Create a demo dataset with synthetic real and fake images"""
    print("[*] Creating demo dataset with synthetic images...")
    
    if max_samples is None:
        max_samples = 1000
    
    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    def assign_split(i, total):
        val_fraction, test_fraction = 0.15, 0.15
        frac = i / total
        if frac < (1 - val_fraction - test_fraction): return "train"
        if frac < (1 - test_fraction): return "val"
        return "test"
    
    def create_real_image():
        # Create a more natural-looking image
        img = Image.new('RGB', (128, 128), color=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
        draw = ImageDraw.Draw(img)
        
        # Add some random shapes to make it look more natural
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, 100), random.randint(0, 100)
            x2, y2 = random.randint(x1+10, 128), random.randint(y1+10, 128)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
        
        # Add some noise
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return img
    
    def create_fake_image():
        # Create an image with obvious AI artifacts
        img = Image.new('RGB', (128, 128), color=(random.randint(80, 180), random.randint(80, 180), random.randint(80, 180)))
        draw = ImageDraw.Draw(img)
        
        # Add geometric patterns that look AI-generated
        for i in range(5):
            x, y = random.randint(0, 100), random.randint(0, 100)
            size = random.randint(10, 30)
            color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
            draw.ellipse([x, y, x+size, y+size], fill=color, outline=(255, 255, 255))
        
        # Add unnatural patterns
        for i in range(0, 128, 8):
            draw.line([(i, 0), (i, 128)], fill=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)), width=1)
        
        return img
    
    # Create images
    total_samples = max_samples * 2  # real + fake
    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}
    
    for i in tqdm(range(total_samples), desc="Creating demo images"):
        is_real = i % 2 == 0
        img_type = "real" if is_real else "fake"
        split = assign_split(i // 2, max_samples)
        
        img = create_real_image() if is_real else create_fake_image()
        fname = output_path / split / img_type / f"{img_type}_{i//2:06d}.jpg"
        img.save(fname, quality=95)
        counts[split][img_type] += 1
    
    # Write meta
    import json
    meta = {
        "dataset": "Demo Synthetic",
        "source": "Generated synthetic images for demo",
        "total": total_samples,
        "real": {split: counts[split]["real"] for split in ["train", "val", "test"]},
        "fake": {split: counts[split]["fake"] for split in ["train", "val", "test"]},
        "image_size": 128,
        "note": "Synthetic demo dataset for training demonstration"
    }
    (output_path / "meta.json").write_text(json.dumps(meta, indent=2))
    
    print(f"\n[OK] Demo dataset created:")
    print(f"     Train: {counts['train']['real']} real + {counts['train']['fake']} fake")
    print(f"     Val:   {counts['val']['real']}   real + {counts['val']['fake']}   fake")
    print(f"     Test:  {counts['test']['real']}  real + {counts['test']['fake']}  fake")
    print(f"     Location: {output_path.resolve()}")
    
    return str(output_path)


def download_cifake(
    output_dir:   str = "data",
    max_samples:  Optional[int] = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
):
    """
    Download CIFAKE and organize into train/val/test splits.

    Final structure:
        data/
          train/ real/ fake/
          val/   real/ fake/
          test/  real/ fake/
          meta.json
    """
    print("\n[*] Downloading CIFAKE dataset from HuggingFace...")
    print("    ~120 MB download — real photos + AI-generated images")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements.txt first."
        ) from exc

    print("[*] Fetching real CIFAKE images (this may take a few minutes)...")
    try:
        ds = load_dataset("yanbax/CIFAKE_autotrain_compatible", split="train")
    except Exception as exc:
        raise RuntimeError(
            "Could not download CIFAKE from HuggingFace. Check your network, or use "
            "--dataset custom --data_dir <folder> with train/val/test real/fake folders."
        ) from exc

    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    label_feature = ds.features.get("label")
    label_names = [name.lower() for name in getattr(label_feature, "names", [])]

    def class_name(item):
        label = item["label"]
        if label_names and isinstance(label, int):
            return label_names[label]
        return str(label).lower()

    # Split the public CIFAKE train pool into train/val/test for this project.
    all_items = list(ds)
    random.shuffle(all_items)

    real_items = [x for x in all_items if class_name(x) == "real"]
    fake_items = [x for x in all_items if class_name(x) == "fake"]

    n = min(len(real_items), len(fake_items))
    if max_samples:
        n = min(n, max_samples)

    real_items = real_items[:n]
    fake_items = fake_items[:n]

    print(f"[*] Using {n} real + {n} fake images ({n*2} total)")

    def assign_split(i, total):
        frac = i / total
        if frac < (1 - val_fraction - test_fraction): return "train"
        if frac < (1 - test_fraction):                return "val"
        return "test"

    def save_items(items, label_name):
        counts = {"train": 0, "val": 0, "test": 0}
        total  = len(items)
        for i, item in enumerate(tqdm(items, desc=f"Saving {label_name}", ncols=70)):
            split = assign_split(i, total)
            img   = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            # Upscale 32×32 → 128×128 with bicubic for better features
            img = img.resize((128, 128), Image.BICUBIC).convert("RGB")
            fname = output_path / split / label_name / f"{label_name}_{i:06d}.jpg"
            img.save(fname, quality=95)
            counts[split] += 1
        return counts

    print("[*] Saving real images...")
    real_counts = save_items(real_items, "real")
    print("[*] Saving fake images...")
    fake_counts = save_items(fake_items, "fake")

    # Write meta
    import json
    meta = {
        "dataset":    "CIFAKE",
        "source":     "HuggingFace: cifake-real-and-ai-generated-synthetic-images",
        "total":      n * 2,
        "real":       real_counts,
        "fake":       fake_counts,
        "image_size": 128,
        "note":       "Images upscaled from 32x32 to 128x128 (bicubic)"
    }
    (output_path / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[OK] CIFAKE downloaded and organized:")
    print(f"     Train: {real_counts['train']} real + {fake_counts['train']} fake")
    print(f"     Val:   {real_counts['val']}   real + {fake_counts['val']}   fake")
    print(f"     Test:  {real_counts['test']}  real + {fake_counts['test']}  fake")
    print(f"     Location: {output_path.resolve()}")

    return str(output_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",   default="data")
    p.add_argument("--max_samples",  type=int, default=None)
    args = p.parse_args()
    download_cifake(args.output_dir, max_samples=args.max_samples)
