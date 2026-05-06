"""
Training Configuration
======================
Central config using dataclasses — override via YAML or CLI arguments.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    train_dir:       str = "data/train"
    val_dir:         str = "data/val"
    test_dir:        str = "data/test"
    image_size:      int = 224
    num_frames:      int = 16          # video frames to sample
    frame_stride:    int = 4           # sample every N frames
    audio_sr:        int = 16000       # audio sample rate
    n_mels:          int = 128
    max_audio_len:   int = 48000       # 3 seconds at 16kHz
    num_workers:     int = 8
    pin_memory:      bool = True
    # Augmentation
    aug_prob:        float = 0.5
    jpeg_quality:    List[int] = field(default_factory=lambda: [25, 95])
    use_mixup:       bool = False
    mixup_alpha:     float = 0.2


@dataclass
class ModelConfig:
    # Image
    image_backbone:   str = "efficientnet_b4"
    image_pretrained: bool = True
    # Video
    video_embed_dim:  int = 512
    video_depth:      int = 6
    video_heads:      int = 8
    # Audio
    audio_pretrained: str = "facebook/wav2vec2-base"
    use_wav2vec:      bool = True


@dataclass
class TrainConfig:
    # Basic
    epochs:           int = 30
    batch_size:       int = 32
    grad_accum_steps: int = 2
    # Optimizer
    lr:               float = 2e-4
    weight_decay:     float = 0.01
    # Scheduler
    warmup_epochs:    int = 3
    scheduler:        str = "cosine"    # cosine | step | plateau
    # Loss
    loss_alpha:       float = 0.5
    loss_beta:        float = 0.3
    loss_gamma:       float = 0.2
    label_smoothing:  float = 0.1
    # Regularization
    dropout:          float = 0.3
    ema_decay:        float = 0.999     # exponential moving average
    # Precision
    mixed_precision:  bool = True       # BF16 / FP16
    # Curriculum
    curriculum:       bool = True
    easy_epochs:      int = 10
    medium_epochs:    int = 10
    hard_epochs:      int = 10
    # Adversarial training
    adv_training:     bool = False
    adv_eps:          float = 8.0 / 255
    adv_steps:        int = 7
    # Checkpointing
    save_dir:         str = "checkpoints"
    save_top_k:       int = 3
    monitor:          str = "val_auc"


@dataclass
class InferenceConfig:
    image_model_path:  Optional[str] = None
    video_model_path:  Optional[str] = None
    audio_model_path:  Optional[str] = None
    ensemble_path:     Optional[str] = None
    device:            str = "cuda"
    batch_size:        int = 16
    use_tta:           bool = True       # test-time augmentation
    tta_n:             int = 8
    threshold_fake:    float = 0.6
    threshold_suspicious: float = 0.3
    # Optimization
    use_tensorrt:      bool = False
    use_onnx:          bool = False
    quantize_int8:     bool = False


@dataclass
class APIConfig:
    host:              str = "0.0.0.0"
    port:              int = 8000
    workers:           int = 4
    max_file_size_mb:  int = 500
    rate_limit:        int = 100          # req / minute
    redis_url:         str = "redis://localhost:6379"
    secret_key:        str = "change-me-in-production"
    allowed_origins:   List[str] = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api:       APIConfig       = field(default_factory=APIConfig)
    project:   str = "deepfake-detector"
    run_name:  str = "exp-001"
    seed:      int = 42


# Default singleton
DEFAULT_CONFIG = Config()
