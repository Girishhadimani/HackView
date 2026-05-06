"""
Audio Deepfake Detector
=======================
Architecture:
  - wav2vec 2.0 (fine-tuned last 4 layers) → 1024-dim temporal features
  - Mel Spectrogram CNN (ResNet-18 on log-mel) → 512-dim
  - Prosody Head (F0, energy, phone boundaries) → 128-dim
  → MLP classifier → fake probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Mel Spectrogram CNN
# ---------------------------------------------------------------------------

class MelSpectrogramCNN(nn.Module):
    """
    ResNet-18-style CNN operating on log-Mel spectrograms.
    Input: (B, 1, n_mels, T) — single-channel spectrogram image
    Output: (B, 512)
    """

    def __init__(self, n_mels: int = 128, out_dim: int = 512):
        super().__init__()

        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = block(32, 64,  stride=1)
        self.layer2 = block(64, 128, stride=2)
        self.layer3 = block(128, 256, stride=2)
        self.layer4 = block(256, 512, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.proj   = nn.Linear(512, out_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Prosody Feature Head
# ---------------------------------------------------------------------------

class ProsodyHead(nn.Module):
    """
    Analyzes F0 contour, energy envelope, and voiced/unvoiced patterns.
    Detects unnatural prosody signatures from TTS / voice-cloning systems.
    Input: (B, T, prosody_features) — pre-extracted prosody features
    Output: (B, 128)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        # prosody_features: [F0, energy, voiced_prob, spectral_centroid]
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)           # (B, T, 2H)
        return self.head(out.mean(dim=1))  # (B, 128)


# ---------------------------------------------------------------------------
# wav2vec 2.0 Encoder
# ---------------------------------------------------------------------------

class Wav2VecEncoder(nn.Module):
    """
    Wraps HuggingFace wav2vec 2.0 — fine-tunes only the last 4 transformer layers.
    Falls back to a simple 1D-CNN if transformers not available.
    """

    def __init__(self, out_dim: int = 1024, pretrained_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.out_dim = out_dim

        if HAS_TRANSFORMERS:
            self.model     = Wav2Vec2Model.from_pretrained(pretrained_name)
            self.use_w2v   = True
            # Freeze all layers except last 4 transformer layers
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            for layer in self.model.encoder.layers[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            self.proj = nn.Linear(self.model.config.hidden_size, out_dim)
        else:
            # Fallback: 1D dilated CNN residual blocks
            self.use_w2v = False
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 64,  kernel_size=10, stride=5, padding=2),  nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv1d(256, out_dim, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            )
            self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T_samples) → (B, out_dim)"""
        if self.use_w2v:
            out = self.model(waveform).last_hidden_state  # (B, T', H)
            out = out.mean(dim=1)                          # (B, H)
            return self.proj(out)
        else:
            x = waveform.unsqueeze(1)                      # (B, 1, T)
            x = self.cnn(x).mean(dim=-1)                   # (B, out_dim)
            return self.proj(x)


# ---------------------------------------------------------------------------
# Main Audio Detector
# ---------------------------------------------------------------------------

class AudioDeepfakeDetector(nn.Module):
    """
    Full audio deepfake detector.

    Inputs:
      waveform     : (B, T_samples) — raw audio at 16kHz
      mel          : (B, 1, n_mels, T_frames) — log-Mel spectrogram
      prosody_feats: (B, T_prosody, 4) — [F0, energy, voiced, centroid]

    Output: dict with 'score' ∈ [0, 1]
    """

    W2V_DIM     = 1024
    MEL_DIM     = 512
    PROSODY_DIM = 128
    FUSION_DIM  = W2V_DIM + MEL_DIM + PROSODY_DIM  # 1664

    def __init__(self, use_wav2vec: bool = True, pretrained_w2v: str = "facebook/wav2vec2-base"):
        super().__init__()

        # wav2vec 2.0 branch
        if use_wav2vec and HAS_TRANSFORMERS:
            self.wav2vec = Wav2VecEncoder(out_dim=self.W2V_DIM, pretrained_name=pretrained_w2v)
        else:
            self.wav2vec = Wav2VecEncoder(out_dim=self.W2V_DIM)  # fallback CNN

        # Mel spectrogram CNN
        self.mel_cnn = MelSpectrogramCNN(n_mels=128, out_dim=self.MEL_DIM)

        # Prosody GRU
        self.prosody = ProsodyHead(input_dim=4, hidden_dim=self.PROSODY_DIM // 2)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.FUSION_DIM),
            nn.Linear(self.FUSION_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        waveform:      Optional[torch.Tensor] = None,
        mel:           Optional[torch.Tensor] = None,
        prosody_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        parts = []

        # wav2vec branch
        if waveform is not None:
            w2v = self.wav2vec(waveform)    # (B, 1024)
        else:
            B = (mel or prosody_feats).shape[0]
            w2v = torch.zeros(B, self.W2V_DIM, device=(mel if mel is not None else prosody_feats).device)
        parts.append(w2v)

        # Mel spectrogram branch
        if mel is not None:
            mel_feat = self.mel_cnn(mel)    # (B, 512)
        else:
            mel_feat = torch.zeros(waveform.shape[0], self.MEL_DIM, device=waveform.device)
        parts.append(mel_feat)

        # Prosody branch
        if prosody_feats is not None:
            pros = self.prosody(prosody_feats)  # (B, 128)
        else:
            B = waveform.shape[0] if waveform is not None else mel.shape[0]
            dev = waveform.device if waveform is not None else mel.device
            pros = torch.zeros(B, self.PROSODY_DIM, device=dev)
        parts.append(pros)

        fused = torch.cat(parts, dim=1)          # (B, 1664)
        logit = self.fusion(fused)
        score = torch.sigmoid(logit)

        return {
            "score":      score,
            "logit":      logit,
            "w2v_feats":  w2v,
            "mel_feats":  mel_feat,
            "pros_feats": pros,
        }
