"""
Full Inference Pipeline
========================
Orchestrates image / video / audio detectors + ensemble to produce
a complete AuthenticityReport for any uploaded media.

Supports: images (jpg/png/webp), videos (mp4/mov/avi), audio (wav/mp3/flac)
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from backend.models.image_detector import DinoV2DeepfakeDetector, ImageDeepfakeDetector, LightImageDetector
from backend.models.video_detector import VideoDeepfakeDetector
from backend.models.audio_detector import AudioDeepfakeDetector
from backend.models.ensemble import DeepfakeEnsemble, score_to_verdict
from backend.inference.preprocessor import (
    preprocess_image, extract_video_frames, extract_audio_features, FaceDetector
)
from backend.utils.scorer import AuthenticityScorer, AuthenticityReport
from backend.utils.gradcam import GradCAM, overlay_heatmap

import cv2
import base64
import io


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class DeepfakeDetectionPipeline:
    """
    Main inference pipeline.

    Usage:
        pipeline = DeepfakeDetectionPipeline(device="cuda")
        report   = pipeline.analyze_bytes(file_bytes, filename="photo.jpg")
    """

    def __init__(
        self,
        device:             str = "auto",
        image_ckpt:         Optional[str] = None,
        video_ckpt:         Optional[str] = None,
        audio_ckpt:         Optional[str] = None,
        ensemble_ckpt:      Optional[str] = None,
        num_frames:         int = 16,
        image_size:         int = 224,
        image_model_arch:    str = "full",
        demo_mode:          bool = True,   # if True, returns plausible mock scores when no weights loaded
    ):
        self.demo_mode  = demo_mode
        self.num_frames = num_frames
        self.image_size = image_size
        self.image_model_arch = image_model_arch

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Pipeline] Device: {self.device}")

        # ── Models ────────────────────────────────────────────
        if image_model_arch == "light":
            self.image_model = LightImageDetector(pretrained=False).to(self.device)
            with torch.no_grad():
                self.image_model(torch.zeros(1, 3, image_size, image_size, device=self.device))
        elif image_model_arch == "dinov2-large":
            self.image_model = DinoV2DeepfakeDetector(
                pretrained=True,
                freeze_backbone=False,
            ).to(self.device)
        else:
            self.image_model = ImageDeepfakeDetector(pretrained=(image_ckpt is None)).to(self.device)
        self.video_model = VideoDeepfakeDetector(num_frames=num_frames, depth=4, pretrained=False).to(self.device)
        self.audio_model = AudioDeepfakeDetector(use_wav2vec=False).to(self.device)
        self.ensemble    = DeepfakeEnsemble().to(self.device)

        # Load checkpoints if provided
        for model, ckpt, name in [
            (self.image_model, image_ckpt,    "image"),
            (self.video_model, video_ckpt,    "video"),
            (self.audio_model, audio_ckpt,    "audio"),
        ]:
            if ckpt and Path(ckpt).exists():
                state = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(state.get("model_state", state))
                print(f"[Pipeline] Loaded {name} checkpoint: {ckpt}")

        self.image_model.eval()
        self.video_model.eval()
        self.audio_model.eval()
        self.ensemble.eval()

        self.face_detector = FaceDetector()
        self.scorer        = AuthenticityScorer()

    # ------------------------------------------------------------------
    # Media type detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_media_type(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXTS:  return "image"
        if ext in VIDEO_EXTS:  return "video"
        if ext in AUDIO_EXTS:  return "audio"
        return "unknown"

    # ------------------------------------------------------------------
    # Image analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _analyze_image(self, image_bytes: bytes) -> Dict:
        tensor, image_np, metadata = preprocess_image(image_bytes, size=self.image_size)
        tensor = tensor.to(self.device)

        out = self.image_model(tensor)
        model_score = float(out["score"].item())
        forensic = self._analyze_forensic_cues(image_np, metadata)
        score = max(model_score, forensic["score"])

        # Grad-CAM heatmap (needs grad)
        heatmap_b64 = None
        try:
            target_layer = self.image_model.backbone.blocks[-1]
            cam_gen = GradCAM(self.image_model, target_layer)
            self.image_model.zero_grad()
            with torch.enable_grad():
                t2 = tensor.clone().requires_grad_(True)
                hm = cam_gen.generate(t2)
            overlay = overlay_heatmap(image_np, hm)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
            cam_gen.remove_hooks()
        except Exception:
            pass

        confidence  = max(float(abs(score - 0.5) * 2), forensic["confidence"])
        freq_score  = float(out.get("freq_features", torch.tensor([score * 0.9])).abs().mean().item())
        art_score   = float(out.get("artifact_features", torch.tensor([score * 0.8])).abs().mean().item())

        return {
            "score":        score,
            "confidence":   confidence,
            "model_score":  model_score,
            "forensic_score": forensic["score"],
            "forensic_cues": forensic["cues"],
            "freq_score":   min(freq_score * 0.1, 1.0),  # scale to [0,1]
            "artifact_score": min(art_score * 0.1, 1.0),
            "metadata":     metadata,
            "heatmap_b64":  heatmap_b64,
        }

    def _analyze_forensic_cues(self, image_np: np.ndarray, metadata: Dict) -> Dict:
        """Conservative image forensics used as a second opinion for face deepfakes."""
        cues = []
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Frequency/noise signal: synthetic or recompressed images often have uneven residuals.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(edges.mean() / 255.0)

        blockiness = 0.0
        if h > 16 and w > 16:
            vertical = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1))[:, 7::8])
            horizontal = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0))[7::8, :])
            baseline = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1))) + 1e-6
            blockiness = float(((vertical + horizontal) / 2.0) / baseline)
            if blockiness > 1.18:
                cues.append("JPEG/block-grid recompression pattern")

        if edge_density > 0.13 and lap_var < 650:
            cues.append("soft facial texture with sharp compositing edges")

        faces = self.face_detector.detect(image_np)
        if faces:
            face = max(faces, key=lambda f: f["w"] * f["h"])
            x, y = max(face["x"], 0), max(face["y"], 0)
            fw, fh = max(face["w"], 1), max(face["h"], 1)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            pad = int(max(fw, fh) * 0.22)
            rx1, ry1 = max(0, x - pad), max(0, y - pad)
            rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)

            face_roi = image_np[y:y2, x:x2]
            ring = image_np[ry1:ry2, rx1:rx2].copy()
            ring[max(0, y - ry1):max(0, y2 - ry1), max(0, x - rx1):max(0, x2 - rx1)] = 0
            ring_mask = np.any(ring > 0, axis=2)

            if face_roi.size and ring_mask.any():
                face_lab = cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB).reshape(-1, 3)
                ring_lab = cv2.cvtColor(ring, cv2.COLOR_RGB2LAB)[ring_mask]
                color_delta = float(np.linalg.norm(face_lab.mean(axis=0) - ring_lab.mean(axis=0)) / 255.0)
                if color_delta > 0.18:
                    cues.append("face/background color tone mismatch")

                face_noise = float(cv2.Laplacian(cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
                ring_gray = cv2.cvtColor(ring, cv2.COLOR_RGB2GRAY)
                ring_noise = float(cv2.Laplacian(ring_gray[ring_mask].reshape(-1, 1), cv2.CV_64F).var()) if ring_mask.any() else 0.0
                if abs(np.log1p(face_noise) - np.log1p(ring_noise)) > 0.75:
                    cues.append("face texture/noise does not match surrounding region")

            if face.get("confidence", 1.0) < 0.82:
                cues.append("low-confidence face geometry")

        if len(metadata.get("exif_anomalies", [])) >= 2:
            cues.append("missing camera provenance metadata")

        score = min(0.25 + 0.15 * len(cues), 0.92)
        if len(cues) >= 3:
            score = max(score, 0.72)
        elif len(cues) == 2:
            score = max(score, 0.58)

        return {
            "score": float(score if cues else 0.0),
            "confidence": float(min(0.25 + 0.12 * len(cues), 0.8)),
            "cues": cues,
        }

    # ------------------------------------------------------------------
    # Video analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _analyze_video(self, video_bytes: bytes) -> Dict:
        frames, meta = extract_video_frames(video_bytes, num_frames=self.num_frames, size=self.image_size)
        frames = frames.to(self.device)

        out = self.video_model(frames)
        score       = float(out["score"].item())
        consistency = float(out["consistency_score"].item())

        return {
            "score":              score,
            "confidence":         float(abs(score - 0.5) * 2),
            "consistency_score":  consistency,
            "lip_sync_score":     0.0,   # requires audio extraction
            "video_meta":         meta,
        }

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _analyze_audio(self, audio_bytes: bytes) -> Dict:
        feats = extract_audio_features(audio_bytes)
        if not feats:
            return {"score": 0.0, "confidence": 0.0}

        mel = feats.get("mel").to(self.device) if feats.get("mel") is not None else None
        wav = feats.get("waveform").to(self.device) if feats.get("waveform") is not None else None

        out   = self.audio_model(waveform=wav, mel=mel)
        score = float(out["score"].item())
        return {"score": score, "confidence": float(abs(score - 0.5) * 2)}

    # ------------------------------------------------------------------
    # Demo / mock mode (for UI demos without trained weights)
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_scores(media_type: str, file_hash: str) -> Dict:
        """Deterministic pseudo-random scores based on file hash for demo."""
        import random
        rng = random.Random(int(file_hash[:8], 16))

        base = rng.uniform(0.0, 1.0)
        return {
            "image_score":    round(base + rng.uniform(-0.1, 0.1), 3) if media_type in ("image", "video") else None,
            "video_score":    round(base + rng.uniform(-0.15, 0.15), 3) if media_type == "video" else None,
            "audio_score":    round(base + rng.uniform(-0.2, 0.2), 3) if media_type == "audio" else None,
            "consistency":    round(rng.uniform(0.0, 1.0), 3),
            "lip_sync":       round(rng.uniform(0.0, 0.5), 3),
            "metadata_score": round(rng.uniform(0.0, 0.3), 3),
        }

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def analyze_bytes(
        self,
        file_bytes: bytes,
        filename:   str = "upload",
    ) -> AuthenticityReport:
        t_start    = time.time()
        media_type = self.detect_media_type(filename)
        file_hash  = hashlib.sha256(file_bytes).hexdigest()

        if self.demo_mode:
            mock = self._mock_scores(media_type, file_hash)
            report = self.scorer.compute(
                image_score=mock["image_score"],
                video_score=mock["video_score"],
                audio_score=mock["audio_score"],
                consistency_score=mock["consistency"],
                lip_sync_score=mock["lip_sync"],
                metadata_score=mock["metadata_score"],
                processing_ms=int((time.time() - t_start) * 1000),
                media_hash=file_hash,
            )
            return report

        # ── Real inference ─────────────────────────────────────
        image_result = video_result = audio_result = None
        heatmap_b64  = None

        if media_type == "image":
            image_result = self._analyze_image(file_bytes)
            heatmap_b64  = image_result.get("heatmap_b64")

        elif media_type == "video":
            image_bytes = self._first_frame_bytes(file_bytes)
            if image_bytes:
                image_result = self._analyze_image(image_bytes)
            video_result = self._analyze_video(file_bytes)

        elif media_type == "audio":
            audio_result = self._analyze_audio(file_bytes)

        # ── Build report ───────────────────────────────────────
        meta = (image_result or {}).get("metadata", {})

        report = self.scorer.compute(
            image_score=image_result["score"]      if image_result else None,
            image_conf=image_result["confidence"]  if image_result else 0.0,
            image_artifacts=image_result.get("forensic_cues", []) if image_result else [],
            video_score=video_result["score"]      if video_result else None,
            video_conf=video_result["confidence"]  if video_result else 0.0,
            consistency_score=video_result.get("consistency_score", 0.0) if video_result else 0.0,
            lip_sync_score=video_result.get("lip_sync_score", 0.0) if video_result else 0.0,
            audio_score=audio_result["score"]      if audio_result else None,
            audio_conf=audio_result["confidence"]  if audio_result else 0.0,
            metadata_score=len(meta.get("exif_anomalies", [])) * 0.1,
            exif_anomalies=meta.get("exif_anomalies", []),
            processing_ms=int((time.time() - t_start) * 1000),
            media_hash=file_hash,
            heatmap_url=heatmap_b64,
        )
        return report

    @staticmethod
    def _first_frame_bytes(video_bytes: bytes, tmp_path: str = "/tmp/_ff.mp4") -> Optional[bytes]:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return bytes(buf)
