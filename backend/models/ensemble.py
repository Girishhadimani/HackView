"""
Ensemble Meta-Learner
=====================
Combines image, video, and audio detection scores with EXIF metadata signals
into a final authenticity verdict via a Gradient Boosting + MLP ensemble.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Neural Meta-Learner (MLP)
# ---------------------------------------------------------------------------

class MetaMLP(nn.Module):
    """
    MLP meta-learner that takes all signals and outputs a fused fake probability.

    Input features (38-dim):
      [0]  image_score
      [1]  video_score
      [2]  audio_score
      [3]  image_confidence
      [4]  video_confidence
      [5]  audio_confidence
      [6]  temporal_consistency_score
      [7]  lip_sync_score
      [8]  freq_anomaly_score
      [9]  artifact_score
      [10] metadata_score  (EXIF / provenance)
      [11..37] reserved / future signals
    """

    INPUT_DIM = 38

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Ensemble Model
# ---------------------------------------------------------------------------

class DeepfakeEnsemble(nn.Module):
    """
    Final ensemble that fuses all modality predictions.

    Usage:
        ensemble = DeepfakeEnsemble()
        features = ensemble.build_feature_vector(
            image_score=0.7, video_score=0.8, audio_score=0.6, ...
        )
        result = ensemble.predict(features)
    """

    # Modality weights for simple weighted average baseline
    WEIGHTS = {
        "image":    0.30,
        "video":    0.35,
        "audio":    0.15,
        "metadata": 0.10,
        "freq":     0.10,
    }

    def __init__(self):
        super().__init__()
        self.mlp = MetaMLP(input_dim=MetaMLP.INPUT_DIM)
        self.gbm = None          # fitted sklearn GBM (saved/loaded separately)
        self.gbm_fitted = False

    # ------------------------------------------------------------------
    # Feature vector construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_feature_vector(
        image_score:          float = 0.0,
        image_confidence:     float = 0.0,
        video_score:          float = 0.0,
        video_confidence:     float = 0.0,
        audio_score:          float = 0.0,
        audio_confidence:     float = 0.0,
        temporal_consistency: float = 0.0,
        lip_sync_score:       float = 0.0,
        freq_anomaly_score:   float = 0.0,
        artifact_score:       float = 0.0,
        metadata_score:       float = 0.0,
        extra_signals:        Optional[List[float]] = None,
    ) -> np.ndarray:
        """Build a fixed-length numpy feature vector for the meta-learner."""
        base = [
            image_score, image_confidence,
            video_score, video_confidence,
            audio_score, audio_confidence,
            temporal_consistency,
            lip_sync_score,
            freq_anomaly_score,
            artifact_score,
            metadata_score,
        ]
        if extra_signals:
            base.extend(extra_signals[:27])
        # Pad to 38 dims
        base = base + [0.0] * (MetaMLP.INPUT_DIM - len(base))
        return np.array(base[:MetaMLP.INPUT_DIM], dtype=np.float32)

    # ------------------------------------------------------------------
    # Weighted average baseline score
    # ------------------------------------------------------------------

    @staticmethod
    def weighted_score(
        image_score:  float = 0.0,
        video_score:  float = 0.0,
        audio_score:  float = 0.0,
        freq_score:   float = 0.0,
        meta_score:   float = 0.0,
        image_avail:  bool = True,
        video_avail:  bool = True,
        audio_avail:  bool = True,
    ) -> float:
        """Compute a simple weighted average (used when MLP not yet trained)."""
        w = DeepfakeEnsemble.WEIGHTS
        total_w = (w["image"] * int(image_avail) +
                   w["video"] * int(video_avail) +
                   w["audio"] * int(audio_avail) +
                   w["freq"] + w["metadata"])
        score = (
            w["image"]    * image_score  * int(image_avail) +
            w["video"]    * video_score  * int(video_avail) +
            w["audio"]    * audio_score  * int(audio_avail) +
            w["freq"]     * freq_score   +
            w["metadata"] * meta_score
        ) / (total_w + 1e-8)
        return float(score)

    # ------------------------------------------------------------------
    # Neural forward pass
    # ------------------------------------------------------------------

    def forward(self, feature_vec: torch.Tensor) -> torch.Tensor:
        """feature_vec: (B, 38) → (B, 1)"""
        return self.mlp(feature_vec)

    # ------------------------------------------------------------------
    # Full prediction (combining GBM + MLP)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, feature_vec: np.ndarray) -> Dict:
        """
        feature_vec: (38,) or (B, 38) numpy array
        Returns authenticity score and verdict.
        """
        self.eval()
        if feature_vec.ndim == 1:
            feature_vec = feature_vec[None]  # (1, 38)

        # Neural prediction
        t_vec      = torch.from_numpy(feature_vec).float()
        mlp_score  = self.mlp(t_vec).numpy().flatten()

        # GBM prediction (if trained)
        if self.gbm_fitted and self.gbm is not None:
            gbm_score = self.gbm.predict_proba(feature_vec)[:, 1]
            final     = 0.6 * gbm_score + 0.4 * mlp_score
        else:
            final = mlp_score

        verdicts = []
        for s in final:
            if s >= 0.8:  verdicts.append("FAKE")
            elif s >= 0.6: verdicts.append("SUSPICIOUS")
            elif s >= 0.3: verdicts.append("UNCERTAIN")
            else:          verdicts.append("REAL")

        return {
            "authenticity_score": float(final[0]) if len(final) == 1 else final.tolist(),
            "verdict":            verdicts[0] if len(verdicts) == 1 else verdicts,
            "mlp_score":          float(mlp_score[0]),
            "confidence":         float(abs(final[0] - 0.5) * 2),
        }

    # ------------------------------------------------------------------
    # GBM fit (call after collecting validation predictions)
    # ------------------------------------------------------------------

    def fit_gbm(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, 38) feature matrix
        y: (N,) binary labels — 1 = fake, 0 = real
        """
        if not HAS_SKLEARN:
            print("scikit-learn not available — skipping GBM training.")
            return
        gbm = GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )
        calibrated = CalibratedClassifierCV(gbm, cv=5, method="isotonic")
        calibrated.fit(X, y)
        self.gbm        = calibrated
        self.gbm_fitted = True
        print(f"GBM fitted on {len(y)} samples.")


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------

def score_to_verdict(score: float) -> Dict:
    """Convert raw fake probability to human-readable verdict."""
    if score >= 0.80:
        return {"verdict": "FAKE",       "color": "#ef4444", "icon": "❌", "risk": "HIGH"}
    elif score >= 0.60:
        return {"verdict": "SUSPICIOUS", "color": "#f97316", "icon": "⚠️", "risk": "MEDIUM"}
    elif score >= 0.30:
        return {"verdict": "UNCERTAIN",  "color": "#eab308", "icon": "🔍", "risk": "LOW"}
    else:
        return {"verdict": "AUTHENTIC",  "color": "#22c55e", "icon": "✅", "risk": "NONE"}
