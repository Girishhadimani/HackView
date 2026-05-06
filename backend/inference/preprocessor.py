"""
Media Preprocessor
==================
Handles: image loading/resizing, video frame extraction, audio extraction,
face detection (MediaPipe), EXIF metadata parsing.
"""

import io
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from PIL.ExifTags import TAGS
    HAS_EXIF = True
except ImportError:
    HAS_EXIF = False


# ---------------------------------------------------------------------------
# Image normalization constants (ImageNet)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
    image_bytes: bytes,
    size: int = 224,
) -> Tuple[torch.Tensor, np.ndarray, Dict]:
    """
    Returns:
        tensor    : (3, H, W) normalized for model input
        image_np  : (H, W, 3) uint8 RGB for visualization
        metadata  : EXIF and file info dict
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    metadata = extract_image_metadata(img, image_bytes)

    # Resize
    img_resized = img.resize((size, size), Image.LANCZOS)
    image_np    = np.array(img_resized, dtype=np.uint8)

    # To tensor and normalize
    tensor = TF.to_tensor(img_resized)        # (3, H, W) float [0,1]
    tensor = normalize_tensor(tensor)

    return tensor.unsqueeze(0), image_np, metadata   # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class FaceDetector:
    """MediaPipe-based face detector for extracting face crops."""

    def __init__(self, min_confidence: float = 0.7):
        self.available = HAS_MEDIAPIPE
        if HAS_MEDIAPIPE:
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,   # long-range model
                min_detection_confidence=min_confidence,
            )

    def detect(self, image_np: np.ndarray) -> List[Dict]:
        """
        Returns list of face bounding boxes: {x, y, w, h, confidence}
        """
        if not self.available:
            return []
        rgb  = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if image_np.shape[2] == 3 else image_np
        res  = self.detector.process(rgb)
        H, W = image_np.shape[:2]

        faces = []
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                faces.append({
                    "x":          int(bb.xmin * W),
                    "y":          int(bb.ymin * H),
                    "w":          int(bb.width * W),
                    "h":          int(bb.height * H),
                    "confidence": det.score[0],
                })
        return faces

    def crop_face(self, image_np: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
        """Return the largest detected face crop, with margin."""
        faces = self.detect(image_np)
        if not faces:
            return None
        face  = max(faces, key=lambda f: f["w"] * f["h"])
        H, W  = image_np.shape[:2]
        mx    = int(face["w"] * margin)
        my    = int(face["h"] * margin)
        x1    = max(0, face["x"] - mx)
        y1    = max(0, face["y"] - my)
        x2    = min(W, face["x"] + face["w"] + mx)
        y2    = min(H, face["y"] + face["h"] + my)
        return image_np[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Video preprocessing
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_bytes: bytes,
    num_frames:  int = 16,
    size:        int = 224,
    tmp_path:    str = "/tmp/_dfake_tmp.mp4",
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract uniformly sampled frames from a video.

    Returns:
        frames_tensor: (T, 3, H, W) normalized
        meta:          video metadata dict
    """
    # Write bytes to temp file (OpenCV needs a path)
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)

    cap   = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    meta = {
        "total_frames": total,
        "fps":          fps,
        "width":        w,
        "height":       h,
        "duration_s":   total / max(fps, 1),
    }

    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))

        t = TF.to_tensor(Image.fromarray(frame))
        t = normalize_tensor(t)
        frames.append(t)

    cap.release()
    frames_tensor = torch.stack(frames).unsqueeze(0)  # (1, T, 3, H, W)
    return frames_tensor, meta


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def extract_audio_features(
    audio_bytes: bytes,
    sr:          int = 16000,
    n_mels:      int = 128,
    max_len:     int = 48000,
    tmp_path:    str = "/tmp/_dfake_audio.wav",
) -> Dict:
    """
    Extract mel spectrogram and waveform features from audio bytes.

    Returns dict with 'waveform', 'mel', 'sr'
    """
    if not HAS_LIBROSA:
        return {}

    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    waveform, orig_sr = librosa.load(tmp_path, sr=sr, mono=True)

    # Pad or trim
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    else:
        waveform = waveform[:max_len]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize mel to [-1, 1]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    mel_tensor  = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)
    wav_tensor  = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)              # (1, T)

    return {"waveform": wav_tensor, "mel": mel_tensor, "sr": sr}


# ---------------------------------------------------------------------------
# EXIF / Metadata extraction
# ---------------------------------------------------------------------------

def extract_image_metadata(img: Image.Image, raw_bytes: bytes) -> Dict:
    """Extract EXIF tags and compute file hash."""
    meta  = {"file_hash": hashlib.sha256(raw_bytes).hexdigest()}
    anomalies = []

    if HAS_EXIF:
        try:
            exif_data = img._getexif() or {}
            decoded   = {TAGS.get(k, k): v for k, v in exif_data.items()}
            meta["exif"] = {k: str(v) for k, v in decoded.items() if isinstance(v, (str, int, float))}
            # Check for suspicious absence of common EXIF fields
            for field in ["Make", "Model", "DateTime"]:
                if field not in decoded:
                    anomalies.append(f"Missing EXIF: {field}")
        except Exception:
            anomalies.append("No EXIF data (possible re-save)")

    meta["exif_anomalies"] = anomalies
    return meta
