# DINOv2-large Deepfake Model

This project supports a Hugging Face DINOv2-large image classifier:

- Backbone: `facebook/dinov2-large`
- Head: binary real/fake classifier
- Checkpoint output: `checkpoints/image_dinov2-large_best.pt`

Important: DINOv2-large is a visual feature backbone, not a deepfake detector by itself. Train or fine-tune it on real/fake data before using it for accuracy.

## Fast Head-Only Fine-Tune

Use this first on CPU or a small machine:

```powershell
python run_training.py --dataset custom --data_dir data --skip_download --model_arch dinov2-large --image_size 224 --epochs 3 --batch_size 4 --num_workers 0 --freeze_backbone
```

## Full Fine-Tune

Use this on a CUDA GPU:

```powershell
python run_training.py --dataset custom --data_dir data --skip_download --model_arch dinov2-large --image_size 224 --epochs 10 --batch_size 8 --num_workers 4
```

## Serve DINOv2 Checkpoint

After training:

```powershell
$env:IMAGE_MODEL_ARCH="dinov2-large"
$env:IMAGE_MODEL_PATH="checkpoints/image_dinov2-large_best.pt"
$env:IMAGE_SIZE="224"
$env:DEMO_MODE="false"
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

If `checkpoints/image_dinov2-large_best.pt` exists, the API will prefer it automatically.
