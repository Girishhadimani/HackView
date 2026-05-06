"""
Health Check Routes
"""

from fastapi import APIRouter
from datetime import datetime
import torch
import platform
import psutil

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "gpu":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu":       platform.processor(),
        "ram_gb":    round(psutil.virtual_memory().total / 1e9, 1),
    }


@router.get("/version", summary="API version")
async def version():
    return {"version": "1.0.0", "model": "EfficientNet-B4 + TimeSformer + wav2vec2"}
