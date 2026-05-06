"""
Dataset Loaders for Deepfake Detection
=======================================
Supports: FaceForensics++, DFDC, Celeb-DF, WildDeepFake, ASVspoof, custom directories.

Structure expected:
    data/
      train/
        real/   ← real images / video frames
        fake/   ← manipulated images / video frames
      val/  (same)
      test/ (same)
"""

import os
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ---------------------------------------------------------------------------
# Image Dataset
# ---------------------------------------------------------------------------

def get_image_transforms(split: str = "train", image_size: int = 224) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ImageCompression(quality_range=(25, 95), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(std_range=(0.04, 0.16), p=1.0),
            ], p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.RandomGamma(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class ImageDeepfakeDataset(Dataset):
    """
    Image-level deepfake dataset.
    Expects directory layout: root/real/*.{jpg,png}  root/fake/*.{jpg,png}
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root:       str,
        split:      str = "train",
        image_size: int = 224,
        transform:  Optional[Callable] = None,
        max_per_class: Optional[int] = None,
    ):
        self.root      = Path(root)
        self.transform = transform or get_image_transforms(split, image_size)
        self.samples: List[Tuple[Path, int]] = []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in self.EXTS]
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend((f, label) for f in files)

        random.shuffle(self.samples)
        print(f"[Dataset] {split}: {len(self.samples)} samples "
              f"({sum(1 for _,l in self.samples if l==0)} real, "
              f"{sum(1 for _,l in self.samples if l==1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        return {"image": aug["image"], "label": torch.tensor(label, dtype=torch.float32), "path": str(path)}


# ---------------------------------------------------------------------------
# Video Dataset
# ---------------------------------------------------------------------------

class VideoDeepfakeDataset(Dataset):
    """
    Video-level deepfake dataset.
    Samples N frames uniformly from each video file.
    """

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        root:       str,
        split:      str = "train",
        num_frames: int = 16,
        image_size: int = 224,
        max_per_class: Optional[int] = None,
    ):
        self.root       = Path(root)
        self.num_frames = num_frames
        self.transform  = get_image_transforms(split, image_size)
        self.samples: List[Tuple[Path, int]] = []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in self.VIDEO_EXTS]
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend((f, label) for f in files)

        random.shuffle(self.samples)

    def _sample_frames(self, video_path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=frame)
            frames.append(aug["image"])

        cap.release()
        return torch.stack(frames)  # (T, 3, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        frames = self._sample_frames(path)
        return {
            "frames": frames,
            "label":  torch.tensor(label, dtype=torch.float32),
            "path":   str(path),
        }


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_image_loaders(
    train_dir:    str,
    val_dir:      str,
    test_dir:     str,
    batch_size:   int = 32,
    image_size:   int = 224,
    num_workers:  int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = ImageDeepfakeDataset(train_dir, split="train", image_size=image_size)
    val_ds   = ImageDeepfakeDataset(val_dir,   split="val",   image_size=image_size)
    test_ds  = ImageDeepfakeDataset(test_dir,  split="test",  image_size=image_size)

    common = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common),
    )
