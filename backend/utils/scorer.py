"""
Authenticity Score Engine
==========================
Combines raw model scores with metadata signals into a comprehensive
multi-dimensional authenticity report.
"""

import hashlib
import struct
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Score data models
# ---------------------------------------------------------------------------

@dataclass
class ModalityScore:
    score:       float           # 0=real, 1=fake
    confidence:  float           # 0–1
    available:   bool = True
    artifacts:   List[str] = field(default_factory=list)


@dataclass
class AuthenticityReport:
    """Full authenticity report for a single media item."""

    # Core scores
    authenticity_score: float    # 0=real → 1=fake
    verdict:            str      # AUTHENTIC | UNCERTAIN | SUSPICIOUS | FAKE
    confidence:         float
    risk_level:         str      # NONE | LOW | MEDIUM | HIGH

    # Modality breakdown
    image_score:        ModalityScore = field(default_factory=lambda: ModalityScore(0.0, 0.0, False))
    video_score:        ModalityScore = field(default_factory=lambda: ModalityScore(0.0, 0.0, False))
    audio_score:        ModalityScore = field(default_factory=lambda: ModalityScore(0.0, 0.0, False))

    # Metadata signals
    metadata_score:     float = 0.0
    exif_anomalies:     List[str] = field(default_factory=list)

    # Technical details
    processing_time_ms: int = 0
    media_hash:         str = ""
    analyzed_at:        str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Visualization
    heatmap_url:        Optional[str] = None
    report_url:         Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------

VERDICT_MAP = [
    (0.80, "FAKE",       "HIGH",   "#ef4444", "❌"),
    (0.60, "SUSPICIOUS", "MEDIUM", "#f97316", "⚠️"),
    (0.30, "UNCERTAIN",  "LOW",    "#eab308", "🔍"),
    (0.00, "AUTHENTIC",  "NONE",   "#22c55e", "✅"),
]


def score_to_verdict(score: float) -> Dict:
    for threshold, verdict, risk, color, icon in VERDICT_MAP:
        if score >= threshold:
            return {"verdict": verdict, "risk_level": risk, "color": color, "icon": icon}
    return {"verdict": "AUTHENTIC", "risk_level": "NONE", "color": "#22c55e", "icon": "✅"}


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def compute_confidence(score: float) -> float:
    """Higher confidence when score is far from decision boundary (0.5)."""
    return float(abs(score - 0.5) * 2.0)


# ---------------------------------------------------------------------------
# Artifact detection helper
# ---------------------------------------------------------------------------

def detect_image_artifacts(freq_score: float, srm_score: float, edge_score: float) -> List[str]:
    artifacts = []
    if freq_score > 0.6:   artifacts.append("GAN spectral fingerprint")
    if srm_score  > 0.6:   artifacts.append("manipulation noise residual")
    if edge_score > 0.6:   artifacts.append("blending edge artifact")
    if freq_score > 0.8:   artifacts.append("periodic frequency anomaly")
    return artifacts


def detect_video_artifacts(consistency: float, lip_sync: float) -> List[str]:
    artifacts = []
    if consistency > 0.6:  artifacts.append("temporal flickering")
    if lip_sync    > 0.6:  artifacts.append("lip-sync mismatch (Wav2Lip)")
    if consistency > 0.85: artifacts.append("severe frame discontinuity")
    return artifacts


def detect_audio_artifacts(w2v_score: float, mel_score: float, pros_score: float) -> List[str]:
    artifacts = []
    if w2v_score  > 0.6:   artifacts.append("voice cloning signature")
    if mel_score  > 0.6:   artifacts.append("spectrogram anomaly")
    if pros_score > 0.6:   artifacts.append("unnatural prosody pattern")
    return artifacts


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class AuthenticityScorer:
    """
    Combines all modality scores and metadata into a final AuthenticityReport.

    Weights (from training plan):
      image     0.30
      video     0.35
      audio     0.15
      metadata  0.10
      freq      0.10  ← included in image score
    """

    WEIGHTS = {"image": 0.30, "video": 0.35, "audio": 0.15, "metadata": 0.20}

    def compute(
        self,
        image_score:      Optional[float] = None,
        image_conf:       float = 0.0,
        image_artifacts:  Optional[List[str]] = None,
        video_score:      Optional[float] = None,
        video_conf:       float = 0.0,
        consistency_score: float = 0.0,
        lip_sync_score:   float = 0.0,
        audio_score:      Optional[float] = None,
        audio_conf:       float = 0.0,
        audio_artifacts:  Optional[List[str]] = None,
        metadata_score:   float = 0.0,
        exif_anomalies:   Optional[List[str]] = None,
        processing_ms:    int = 0,
        media_hash:       str = "",
        heatmap_url:      Optional[str] = None,
    ) -> AuthenticityReport:

        # Weighted average over available modalities
        weighted_sum = 0.0
        total_weight = 0.0

        modality_scores = {}

        if image_score is not None:
            modality_scores["image"] = image_score
            weighted_sum  += self.WEIGHTS["image"] * image_score
            total_weight  += self.WEIGHTS["image"]

        if video_score is not None:
            # Boost video score with consistency & lip-sync signals
            enhanced_video = 0.5 * video_score + 0.3 * consistency_score + 0.2 * lip_sync_score
            modality_scores["video"] = enhanced_video
            weighted_sum  += self.WEIGHTS["video"] * enhanced_video
            total_weight  += self.WEIGHTS["video"]

        if audio_score is not None:
            modality_scores["audio"] = audio_score
            weighted_sum  += self.WEIGHTS["audio"] * audio_score
            total_weight  += self.WEIGHTS["audio"]

        # Metadata always contributes
        weighted_sum  += self.WEIGHTS["metadata"] * metadata_score
        total_weight  += self.WEIGHTS["metadata"]

        final_score = weighted_sum / (total_weight + 1e-8)
        confidence  = compute_confidence(final_score)
        verdict_info = score_to_verdict(final_score)

        # Build modality score objects
        img_ms = ModalityScore(
            score=image_score or 0.0,
            confidence=image_conf,
            available=image_score is not None,
            artifacts=image_artifacts or [],
        )
        vid_artifacts = detect_video_artifacts(consistency_score, lip_sync_score)
        vid_ms = ModalityScore(
            score=video_score or 0.0,
            confidence=video_conf,
            available=video_score is not None,
            artifacts=vid_artifacts,
        )
        aud_ms = ModalityScore(
            score=audio_score or 0.0,
            confidence=audio_conf,
            available=audio_score is not None,
            artifacts=audio_artifacts or [],
        )

        return AuthenticityReport(
            authenticity_score=round(final_score, 4),
            verdict=verdict_info["verdict"],
            confidence=round(confidence, 4),
            risk_level=verdict_info["risk_level"],
            image_score=img_ms,
            video_score=vid_ms,
            audio_score=aud_ms,
            metadata_score=round(metadata_score, 4),
            exif_anomalies=exif_anomalies or [],
            processing_time_ms=processing_ms,
            media_hash=media_hash,
            heatmap_url=heatmap_url,
        )
