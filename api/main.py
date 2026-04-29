"""
main.py – FastAPI inference server for CIFAR-100 EfficientNet-B0.

Endpoints:
    GET  /health             – liveness / model status
    POST /predict            – single-image inference
    POST /predict/batch      – batch inference (up to 8 images)
    GET  /metrics            – Prometheus-style counters (optional)

Environment variables:
    MODEL_FP32_PATH    – local path or gs:// URI to the FP32 checkpoint
    MODEL_INT8_PATH    – local path or gs:// URI to the INT8 TorchScript file
    HOST               – bind host (default 0.0.0.0)
    PORT               – bind port (default 8080)
    LOG_LEVEL          – uvicorn log level (default info)
    MAX_BATCH_SIZE     – maximum images per batch request (default 8)
"""

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.inference import (
    ModelRegistry,
    decode_image,
    preprocess,
    preprocess_batch,
    run_inference,
)
from api.profiling import RequestProfiler
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    ProfileBreakdown,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global model registry (populated at startup) ───────────────────────────────
registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once before accepting requests."""
    logger.info("Loading models…")
    try:
        registry.load_all()
        logger.info("Models loaded: %s", registry.loaded_keys)
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        # Allow the server to start so /health can report the failure
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CIFAR-100 Inference Service",
    description=(
        "End-to-end image classification service backed by EfficientNet-B0 "
        "fine-tuned on CIFAR-100. Supports FP32 and INT8 (quantized) variants."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simple in-memory counters for /metrics ─────────────────────────────────────
_counters = {"requests_total": 0, "errors_total": 0, "inference_ms_total": 0.0}


# ── Middleware: request logging ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("%s %s  %d  %.1fms", request.method, request.url.path,
                response.status_code, elapsed)
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok" if registry.loaded_keys else "degraded",
        models_loaded=registry.loaded_keys,
        device=str(registry.device),
    )


@app.get("/metrics", tags=["ops"])
def metrics():
    return {
        "requests_total":     _counters["requests_total"],
        "errors_total":       _counters["errors_total"],
        "avg_inference_ms":   (
            _counters["inference_ms_total"] / max(_counters["requests_total"], 1)
        ),
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    _counters["requests_total"] += 1

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(req.image_b64)
        pil_image   = decode_image(image_bytes)
    except Exception as exc:
        _counters["errors_total"] += 1
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    # Get model
    try:
        model = registry.get(req.model_precision)
    except KeyError as exc:
        _counters["errors_total"] += 1
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profiler = RequestProfiler(
        enabled=req.return_profile,
        use_cuda=(str(registry.device) != "cpu"),
    )

    t_total = time.perf_counter()
    with profiler:
        with profiler.stage("preprocess"):
            tensor = preprocess(pil_image)

        with profiler.stage("forward"):
            # Dynamic Device Routing: If it's INT8, use CPU. Otherwise, use the GPU.
            target_device = "cpu" if req.model_precision == "int8" else registry.device
            predictions_raw, _ = run_inference(model, tensor, target_device)

        with profiler.stage("postprocess"):
            top5 = [
                Prediction(**p) for p in predictions_raw[0]
            ]

    inference_ms = (time.perf_counter() - t_total) * 1000
    _counters["inference_ms_total"] += inference_ms

    profile_out = None
    if req.return_profile:
        bd = profiler.breakdown
        profile_out = ProfileBreakdown(
            preprocess_ms=bd.preprocess_ms,
            forward_ms=bd.forward_ms,
            postprocess_ms=bd.postprocess_ms,
            total_ms=bd.total_ms,
        )

    return PredictResponse(
        predictions=top5,
        model_precision=req.model_precision,
        inference_ms=round(inference_ms, 3),
        profile=profile_out,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest):
    max_batch = int(os.environ.get("MAX_BATCH_SIZE", 8))
    if len(req.images_b64) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.images_b64)} exceeds maximum {max_batch}",
        )

    _counters["requests_total"] += 1

    # Decode all images
    try:
        pil_images = [decode_image(base64.b64decode(b64)) for b64 in req.images_b64]
    except Exception as exc:
        _counters["errors_total"] += 1
        raise HTTPException(status_code=400, detail=f"Invalid image in batch: {exc}") from exc

    try:
        model = registry.get(req.model_precision)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profiler = RequestProfiler(
        enabled=req.return_profile,
        use_cuda=(str(registry.device) != "cpu"),
    )

    t_total = time.perf_counter()
    with profiler:
        with profiler.stage("preprocess"):
            batch_tensor = preprocess_batch(pil_images)

        with profiler.stage("forward"):
            target_device = "cpu" if req.model_precision == "int8" else registry.device
            all_preds, _ = run_inference(model, batch_tensor, target_device)

        with profiler.stage("postprocess"):
            results = []
            for preds_for_image in all_preds:
                results.append(PredictResponse(
                    predictions=[Prediction(**p) for p in preds_for_image],
                    model_precision=req.model_precision,
                    inference_ms=0.0,  # filled below per-image estimate
                    profile=None,
                ))

    total_ms = (time.perf_counter() - t_total) * 1000
    per_image_ms = total_ms / len(pil_images)
    for r in results:
        r.inference_ms = round(per_image_ms, 3)

    _counters["inference_ms_total"] += total_ms

    return BatchPredictResponse(
        results=results,
        batch_size=len(pil_images),
        total_inference_ms=round(total_ms, 3),
    )


# ── Local dev entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8080)),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        reload=False,
    )
