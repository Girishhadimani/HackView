"""
Custom Loss Functions for Deepfake Detection Training
=====================================================
- FocalLoss: handles class imbalance (real >> fake in the wild)
- ContrastiveLoss: pulls real embeddings together, pushes fake apart
- CombinedLoss: α·BCE + β·Focal + γ·Contrastive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Focuses training on hard examples by down-weighting easy negatives.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t  = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":   return loss.mean()
        if self.reduction == "sum":    return loss.sum()
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss on feature embeddings.
    - Same class (both real or both fake): minimize distance
    - Different classes: push apart by at least `margin`
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin    = margin
        self.reduction = reduction

    def forward(
        self,
        emb1: torch.Tensor,    # (B, D)
        emb2: torch.Tensor,    # (B, D)
        labels: torch.Tensor,  # (B,) — 1 if same class, 0 if different
    ) -> torch.Tensor:
        dist = F.pairwise_distance(
            F.normalize(emb1, dim=1),
            F.normalize(emb2, dim=1),
        )
        loss = (labels * dist.pow(2) +
                (1 - labels) * F.relu(self.margin - dist).pow(2))
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss


class LabelSmoothingBCE(nn.Module):
    """BCE with label smoothing to improve calibration."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


class CombinedLoss(nn.Module):
    """
    α·LabelSmoothingBCE + β·FocalLoss + γ·ContrastiveLoss
    Default weights from the training plan.
    """

    def __init__(
        self,
        alpha: float = 0.5,    # BCE weight
        beta:  float = 0.3,    # Focal weight
        gamma: float = 0.2,    # Contrastive weight
        label_smoothing: float = 0.1,
        focal_alpha:     float = 0.25,
        focal_gamma:     float = 2.0,
        contrastive_margin: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.bce_loss   = LabelSmoothingBCE(smoothing=label_smoothing)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.cont_loss  = ContrastiveLoss(margin=contrastive_margin)

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        emb1:    Optional[torch.Tensor] = None,
        emb2:    Optional[torch.Tensor] = None,
        same_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.alpha * self.bce_loss(logits, targets.float())
        loss = loss + self.beta * self.focal_loss(logits, targets.float())

        if emb1 is not None and emb2 is not None and same_class is not None:
            loss = loss + self.gamma * self.cont_loss(emb1, emb2, same_class.float())

        return loss
