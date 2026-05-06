"""
App State — shared singleton to avoid circular imports.
Uses TYPE_CHECKING to avoid loading heavy ML imports at module level.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.inference.pipeline import DeepfakeDetectionPipeline

_pipeline = None


def get_pipeline():
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


def set_pipeline(p):
    global _pipeline
    _pipeline = p
