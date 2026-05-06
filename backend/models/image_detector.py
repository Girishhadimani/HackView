"""
Image Deepfake Detector
=======================
Architecture: EfficientNet-B4 + SRM Filters + Frequency Head + Artifact Head → Ensemble Classifier

Key components:
  - SRMFilter: Steganalysis Rich Model 5×5 kernels to detect manipulation residuals invisible to naked eye
  - FrequencyHead: 2D-FFT + CNN pipeline to detect GAN spectral fingerprints
  - ArtifactHead: Compression & blending edge detection via SRM-filtered features
  - ImageDeepfakeDetector: Full model fusing all three branches with attention-weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# SRM Filter Bank
# ---------------------------------------------------------------------------

class SRMFilter(nn.Module):
    """
    Steganalysis Rich Model high-pass filter bank.
    Three 5×5 kernels designed to extract manipulation noise residuals.
    Weights are fixed (not learned) — they encode forensic priors.
    """

    def __init__(self):
        super().__init__()

        q = [4.0, 12.0, 2.0]

        # HPF kernel
        f1 = np.array([[0,  0,  0,  0, 0],
                       [0, -1,  2, -1, 0],
                       [0,  2, -4,  2, 0],
                       [0, -1,  2, -1, 0],
                       [0,  0,  0,  0, 0]], dtype=np.float32) / q[0]

        # Laplacian of Gaussian
        f2 = np.array([[-1,  2, -2,  2, -1],
                       [ 2, -6,  8, -6,  2],
                       [-2,  8,-12,  8, -2],
                       [ 2, -6,  8, -6,  2],
                       [-1,  2, -2,  2, -1]], dtype=np.float32) / q[1]

        # Simple gradient
        f3 = np.array([[0, 0,  0, 0, 0],
                       [0, 0,  0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0,  0, 0, 0],
                       [0, 0,  0, 0, 0]], dtype=np.float32) / q[2]

        # Stack into (3, 3, 5, 5): 3 output channels, 3 input RGB channels
        srm = np.stack([
            np.stack([f1, f1, f1]),
            np.stack([f2, f2, f2]),
            np.stack([f3, f3, f3]),
        ], axis=0)

        self.register_buffer("weight", torch.from_numpy(srm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=2)


# ---------------------------------------------------------------------------
# Frequency Domain Head
# ---------------------------------------------------------------------------

class FrequencyHead(nn.Module):
    """
    Detects GAN fingerprints via FFT magnitude spectrum analysis.
    GAN models leave periodic spectral artifacts (peaks at regular intervals)
    in the Fourier domain that are invisible in pixel space.
    """

    def __init__(self, input_size: int = 224, output_dim: int = 512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),  # → (B, 128, 8, 8)
        )

        self.fc = nn.Linear(128 * 8 * 8, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2D FFT on each channel
        x_fft = torch.fft.fft2(x.float(), norm="ortho")
        magnitude = torch.abs(x_fft)
        magnitude = torch.fft.fftshift(magnitude)        # zero-freq to center

        # Log-scale & normalize to [0, 1]
        magnitude = torch.log(magnitude + 1e-8)
        mn = magnitude.amin(dim=(-1, -2), keepdim=True)
        mx = magnitude.amax(dim=(-1, -2), keepdim=True)
        magnitude = (magnitude - mn) / (mx - mn + 1e-8)

        feats = self.conv(magnitude)
        feats = feats.flatten(1)
        return self.norm(self.fc(feats))


# ---------------------------------------------------------------------------
# Artifact Detection Head
# ---------------------------------------------------------------------------

class ArtifactHead(nn.Module):
    """
    Detects blending edges, double-JPEG compression artifacts,
    and copy-move traces using SRM-filtered noise residuals.
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.srm = SRMFilter()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),  # → (B, 128, 8, 8)
        )

        self.fc = nn.Linear(128 * 8 * 8, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        srm_out = self.srm(x)
        srm_out = torch.tanh(srm_out)          # clamp residuals
        feats = self.conv(srm_out)
        feats = feats.flatten(1)
        return self.norm(self.fc(feats))


# ---------------------------------------------------------------------------
# Attention-Weighted Fusion
# ---------------------------------------------------------------------------

class FeatureAttention(nn.Module):
    """Channel-wise attention over concatenated multi-branch features."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


# ---------------------------------------------------------------------------
# Main Image Detector
# ---------------------------------------------------------------------------

class ImageDeepfakeDetector(nn.Module):
    """
    Full image deepfake detector.

    Inputs : (B, 3, H, W)  — normalized RGB image (224×224 or larger)
    Outputs: dict with 'score' ∈ [0,1] (1 = fake), plus intermediate features
    """

    SPATIAL_DIM = 1792   # EfficientNet-B4 feature dim
    FREQ_DIM    = 512
    ART_DIM     = 256
    FUSION_DIM  = SPATIAL_DIM + FREQ_DIM + ART_DIM   # 2560

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Spatial backbone ───────────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Forensic heads ─────────────────────────────────────
        self.freq_head     = FrequencyHead(output_dim=self.FREQ_DIM)
        self.artifact_head = ArtifactHead(output_dim=self.ART_DIM)

        # ── Attention + Classifier ─────────────────────────────
        self.attention = FeatureAttention(self.FUSION_DIM)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.FUSION_DIM),
            nn.Linear(self.FUSION_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Spatial branch
        spatial_map = self.backbone(x)                          # (B, 1792, h, w)
        spatial = self.gap(spatial_map).squeeze(-1).squeeze(-1) # (B, 1792)

        # 2. Frequency branch
        freq = self.freq_head(x)          # (B, 512)

        # 3. Artifact branch
        artifact = self.artifact_head(x)  # (B, 256)

        # 4. Fuse
        fused = torch.cat([spatial, freq, artifact], dim=1)   # (B, 2560)
        fused = self.attention(fused)

        # 5. Classify
        logit = self.classifier(fused)
        score = torch.sigmoid(logit)

        return {
            "score":             score,
            "logit":             logit,
            "spatial_features":  spatial,
            "freq_features":     freq,
            "artifact_features": artifact,
            "spatial_map":       spatial_map,   # for Grad-CAM
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict:
        """Inference-only forward (returns Python floats)."""
        self.eval()
        out = self.forward(x)
        score = out["score"].cpu().numpy().flatten()
        return {
            "scores":     score.tolist(),
            "verdict":    ["FAKE" if s > 0.6 else "SUSPICIOUS" if s > 0.3 else "REAL" for s in score],
            "confidence": (np.abs(score - 0.5) * 2).tolist(),
        }

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone feature map (before GAP) — used for Grad-CAM."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Lightweight student model (distillation target — fast inference)
# ---------------------------------------------------------------------------

class LightImageDetector(nn.Module):
    """
    MobileNetV3-Small backbone — for edge / real-time deployment.
    Distilled from the full ImageDeepfakeDetector.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        logit = self.head(feats)
        return {"score": torch.sigmoid(logit), "logit": logit}


# ---------------------------------------------------------------------------
# DINOv2-large model (Hugging Face)
# ---------------------------------------------------------------------------

class DinoV2DeepfakeDetector(nn.Module):
    """
    DINOv2-large visual backbone with a binary deepfake head.

    The backbone comes from Hugging Face model id `facebook/dinov2-large`.
    It must be fine-tuned on real/fake media for deepfake detection; the
    pretrained backbone alone is not a deepfake classifier.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if not pretrained:
            raise ValueError("DINOv2 requires pretrained=True so the Hugging Face backbone can be loaded.")

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use DINOv2: pip install transformers") from exc

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out.last_hidden_state[:, 0]
        logit = self.head(feats)
        return {
            "score": torch.sigmoid(logit),
            "logit": logit,
            "spatial_features": feats,
        }
