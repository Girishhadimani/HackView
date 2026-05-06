"""
FastAPI Main Application
========================
Deepfake Detection API with file upload, async job queue, and health checks.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.api import state as app_state_module
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.openapi.utils import get_openapi

from backend.api.routes.analyze import router as analyze_router
from backend.api.routes.health  import router as health_router
from backend.inference.pipeline import DeepfakeDetectionPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the detection pipeline on startup."""
    print("[*] Starting Deepfake Detection API...")
    dinov2_ckpt = Path("checkpoints/image_dinov2-large_best.pt")
    light_ckpt = Path("checkpoints/image_light_best.pt")
    default_ckpt = dinov2_ckpt if dinov2_ckpt.exists() else light_ckpt
    image_ckpt = os.getenv("IMAGE_MODEL_PATH")
    if not image_ckpt and default_ckpt.exists():
        image_ckpt = str(default_ckpt)

    demo_default = "false" if image_ckpt else "true"
    pipeline = DeepfakeDetectionPipeline(
        device="auto",
        image_ckpt=image_ckpt,
        image_model_arch=os.getenv(
            "IMAGE_MODEL_ARCH",
            "dinov2-large" if image_ckpt and "dinov2" in Path(image_ckpt).name else "light" if image_ckpt else "full",
        ),
        image_size=int(os.getenv("IMAGE_SIZE", "224" if image_ckpt and "dinov2" in Path(image_ckpt).name else "128" if image_ckpt else "224")),
        demo_mode=os.getenv("DEMO_MODE", demo_default).lower() == "true",
    )
    app_state_module.set_pipeline(pipeline)
    print("[OK] Pipeline initialized.")
    yield
    print("[*] Shutting down...")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="🛡️ DeepFake Detector API",
    description=(
        "AI-powered deepfake detection for images, videos, and audio. "
        "Uses EfficientNet-B4 + TimeSformer + wav2vec2 ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────
app.include_router(analyze_router, prefix="/v1", tags=["Analysis"])
app.include_router(health_router,  prefix="/v1", tags=["Health"])

# ── Serve frontend ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_frontend():
        index = FRONTEND_DIR / "index.html"
        return HTMLResponse(content=index.read_text(encoding="utf-8"))

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# ── Global error handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
