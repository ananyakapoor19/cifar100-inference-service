"""
schemas.py – Pydantic request/response models for the inference API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single-image inference request.
    The image must be base64-encoded (standard encoding, no data-URI prefix).
    """
    image_b64: str = Field(..., description="Base64-encoded image bytes (JPEG/PNG)")
    model_precision: str = Field(
        default="fp32",
        pattern="^(fp32|int8)$",
        description="Which model variant to use: 'fp32' or 'int8'",
    )
    return_profile: bool = Field(
        default=False,
        description="If true, include per-stage latency breakdown in the response",
    )


class BatchPredictRequest(BaseModel):
    """Batch inference request (up to 8 images)."""
    images_b64: List[str] = Field(..., min_length=1, max_length=8,
                                  description="List of base64-encoded images")
    model_precision: str = Field(default="fp32", pattern="^(fp32|int8)$")
    return_profile: bool = Field(default=False)


class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class ProfileBreakdown(BaseModel):
    preprocess_ms: float
    forward_ms: float
    postprocess_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    predictions: List[Prediction] = Field(...,
        description="Top-5 predictions sorted by confidence (descending)")
    model_precision: str
    inference_ms: float = Field(..., description="End-to-end server-side latency in ms")
    profile: Optional[ProfileBreakdown] = None


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    batch_size: int
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str
