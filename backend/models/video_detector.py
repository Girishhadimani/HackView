"""
Video Deepfake Detector
=======================
Architecture:
  - Frame-level: EfficientNet-B4 encoder
  - Temporal:    Divided Space-Time Transformer (TimeSformer-style)
  - Consistency: BiLSTM for frame-to-frame coherence
  - Lip-Sync:    Audio-visual cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange, repeat
from typing import Dict, Optional


class DividedSpaceTimeAttention(nn.Module):
    """Temporal attention then Spatial attention (separated for efficiency)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.t_qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.t_proj = nn.Linear(dim, dim)
        self.s_qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.s_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _attn(self, qkv_fn, proj_fn, x):
        B, N, C = x.shape
        qkv = qkv_fn(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(proj_fn(out))

    def forward(self, x, T, HW):
        B = x.shape[0]
        x_t = rearrange(x, "b (t hw) c -> (b hw) t c", t=T, hw=HW)
        x_t = self._attn(self.t_qkv, self.t_proj, x_t)
        x   = x + rearrange(x_t, "(b hw) t c -> b (t hw) c", b=B, hw=HW)
        x_s = rearrange(x, "b (t hw) c -> (b t) hw c", t=T, hw=HW)
        x_s = self._attn(self.s_qkv, self.s_proj, x_s)
        x   = x + rearrange(x_s, "(b t) hw c -> b (t hw) c", b=B, t=T)
        return x


class TimeSformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = DividedSpaceTimeAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, T, HW):
        x = x + self.attn(self.norm1(x), T, HW)
        x = x + self.mlp(self.norm2(x))
        return x


class FrameEncoder(nn.Module):
    """EfficientNet-B4 → patch tokens."""

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone   = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="")
        self.token_proj = nn.Linear(1792, embed_dim)

    def forward(self, frames):   # (B*T, 3, H, W)
        feat = self.backbone(frames)                      # (B*T, 1792, h, w)
        B, C, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)          # (B*T, h*w, 1792)
        return self.token_proj(tokens)                    # (B*T, h*w, D)


class TemporalConsistencyLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):   # (B, T, D) → (B, 1)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


class LipSyncVerifier(nn.Module):
    def __init__(self, visual_dim=256, audio_dim=256, embed_dim=128):
        super().__init__()
        self.v_enc = nn.Sequential(nn.Linear(visual_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.a_enc = nn.Sequential(nn.Linear(audio_dim,  embed_dim), nn.LayerNorm(embed_dim))
        self.cross = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.head  = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, lip_feats, audio_feats):
        v = self.v_enc(lip_feats); a = self.a_enc(audio_feats)
        att, _ = self.cross(v, a, a)
        return self.head(att.mean(dim=1))


class VideoDeepfakeDetector(nn.Module):
    """
    Full video deepfake detector.
    Input:  frames (B, T, 3, H, W), optional audio_feats (B, T, audio_dim)
    Output: dict with 'score' ∈ [0,1]
    """

    def __init__(self, num_frames=16, embed_dim=512, num_heads=8, depth=6, audio_dim=256, pretrained=True):
        super().__init__()
        self.T         = num_frames
        self.embed_dim = embed_dim

        self.frame_encoder     = FrameEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.cls_token         = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks            = nn.ModuleList([TimeSformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm              = nn.LayerNorm(embed_dim)
        self.consistency_lstm  = TemporalConsistencyLSTM(embed_dim)
        self.has_audio         = audio_dim > 0
        if self.has_audio:
            self.lip_sync = LipSyncVerifier(visual_dim=embed_dim, audio_dim=audio_dim)

        extra = 2 if self.has_audio else 1
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + extra, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid()
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _pos_embed(self, T, HW, device):
        total = T * HW + 1
        pos   = torch.zeros(1, total, self.embed_dim, device=device)
        pos_  = torch.arange(total, device=device).unsqueeze(1).float()
        div   = torch.exp(torch.arange(0, self.embed_dim, 2, device=device).float() * (-9.210 / self.embed_dim))
        pos[0, :, 0::2] = torch.sin(pos_ * div)
        pos[0, :, 1::2] = torch.cos(pos_ * div)
        return pos

    def forward(self, frames, audio_feats=None):
        B, T, C, H, W = frames.shape
        flat   = frames.reshape(B * T, C, H, W)
        tokens = self.frame_encoder(flat)             # (B*T, HW, D)
        _, HW, D = tokens.shape
        tokens = tokens.reshape(B, T * HW, D)
        cls    = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        tokens = torch.cat([cls, tokens], dim=1) + self._pos_embed(T, HW, frames.device)

        for block in self.blocks:
            tokens = block(tokens, T=T, HW=HW)
        tokens = self.norm(tokens)

        cls_out      = tokens[:, 0]
        frame_tokens = tokens[:, 1:].reshape(B, T, HW, D).mean(dim=2)
        consist      = self.consistency_lstm(frame_tokens)

        if self.has_audio and audio_feats is not None:
            lip   = self.lip_sync(frame_tokens, audio_feats)
            fused = torch.cat([cls_out, consist, lip], dim=1)
        else:
            fused = torch.cat([cls_out, consist], dim=1)

        score = self.classifier(fused)
        return {"score": score, "consistency_score": consist, "cls_features": cls_out}
