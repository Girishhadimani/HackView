"""
Analysis Routes
===============
POST /v1/analyze  — Upload and analyze media file
GET  /v1/report/{job_id} — Fetch stored report
"""

import uuid
import time

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse

from backend.api.state import get_pipeline
from backend.inference.pipeline import DeepfakeDetectionPipeline

router = APIRouter()

# In-memory job store (use Redis in production)
JOB_STORE: dict = {}

MAX_FILE_SIZE = 500 * 1024 * 1024   # 500 MB


@router.post("/analyze", summary="Analyze media for deepfake content")
async def analyze_media(
    file:     UploadFile = File(..., description="Image, video, or audio file"),
    pipeline: DeepfakeDetectionPipeline = Depends(get_pipeline),
):
    """
    Upload a media file and receive a deepfake authenticity report.

    Supported formats:
    - **Images**: jpg, jpeg, png, bmp, webp
    - **Videos**: mp4, mov, avi, mkv, webm
    - **Audio**:  wav, mp3, flac, ogg, m4a
    """
    # Validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 500MB)")

    filename = file.filename or "upload"
    media_type = pipeline.detect_media_type(filename)
    if media_type == "unknown":
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {filename}")

    # Run detection
    try:
        t0     = time.time()
        report = pipeline.analyze_bytes(content, filename=filename)
        elapsed_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Store result
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = report.to_dict()

    result = report.to_dict()
    result["job_id"]           = job_id
    result["filename"]         = filename
    result["media_type"]       = media_type
    result["processing_time_ms"] = elapsed_ms

    return JSONResponse(content=result)


@router.get("/report/{job_id}", summary="Fetch analysis report by job ID")
async def get_report(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(content=JOB_STORE[job_id])


@router.get("/analyze/url", summary="Analyze media from URL")
async def analyze_url(
    url:      str,
    pipeline: DeepfakeDetectionPipeline = Depends(get_pipeline),
):
    """Analyze media from a public URL (social media link, CDN URL, etc.)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            content  = resp.content
            filename = url.split("/")[-1].split("?")[0] or "download"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    try:
        report = pipeline.analyze_bytes(content, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = report.to_dict()
    result = report.to_dict()
    result["job_id"] = job_id
    result["source_url"] = url
    return JSONResponse(content=result)
