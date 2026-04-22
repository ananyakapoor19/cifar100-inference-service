"""
test_api.py – Integration tests for the FastAPI inference server.

Run with:
    pytest tests/test_api.py -v

The tests use TestClient (synchronous ASGI test client) and mock the ModelRegistry
so no actual model checkpoints are required.
"""

import base64
import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_random_image_b64(width: int = 32, height: int = 32) -> str:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def make_mock_model(num_classes: int = 100) -> MagicMock:
    """Return a mock nn.Module that outputs random logits."""
    mock = MagicMock()
    def forward(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.randn(batch, num_classes)
    mock.side_effect = forward  # side_effect is honoured by MagicMock.__call__
    mock.parameters = lambda: iter([torch.zeros(1)])
    return mock


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a mocked model registry."""
    # Set env vars before importing app
    os.environ["MODEL_FP32_PATH"] = "/fake/fp32.pth"
    os.environ["MODEL_INT8_PATH"] = "/fake/int8.pth"

    # Patch load_all so no files are needed
    with patch("api.inference.ModelRegistry.load_all") as mock_load_all:
        from api.main import app, registry

        # Inject fake models
        registry._models = {
            "fp32": make_mock_model(),
            "int8": make_mock_model(),
        }

        with TestClient(app) as c:
            yield c


# ── Tests: /health ─────────────────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert isinstance(data["models_loaded"], list)
    assert isinstance(data["device"], str)


# ── Tests: /predict ────────────────────────────────────────────────────────────

def test_predict_fp32(client):
    payload = {
        "image_b64":       make_random_image_b64(),
        "model_precision": "fp32",
        "return_profile":  False,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_precision"] == "fp32"
    assert len(data["predictions"]) == 5
    assert data["profile"] is None
    # Confidences sum to ≤ 1 (top-5 subset)
    conf_sum = sum(p["confidence"] for p in data["predictions"])
    assert 0.0 < conf_sum <= 1.0 + 1e-5


def test_predict_int8(client):
    payload = {
        "image_b64":       make_random_image_b64(),
        "model_precision": "int8",
        "return_profile":  False,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["model_precision"] == "int8"


def test_predict_with_profile(client):
    payload = {
        "image_b64":       make_random_image_b64(),
        "model_precision": "fp32",
        "return_profile":  True,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    profile = resp.json().get("profile")
    assert profile is not None
    for key in ("preprocess_ms", "forward_ms", "postprocess_ms", "total_ms"):
        assert key in profile
        assert profile[key] >= 0.0


def test_predict_invalid_base64(client):
    payload = {
        "image_b64": "!!!not_valid_base64!!!",
        "model_precision": "fp32",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400


def test_predict_invalid_precision(client):
    payload = {
        "image_b64":       make_random_image_b64(),
        "model_precision": "bf16",   # not supported
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422   # Pydantic validation error


# ── Tests: /predict/batch ──────────────────────────────────────────────────────

def test_batch_predict_size_1(client):
    payload = {
        "images_b64":      [make_random_image_b64()],
        "model_precision": "fp32",
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["batch_size"] == 1
    assert len(data["results"]) == 1


def test_batch_predict_size_8(client):
    payload = {
        "images_b64":      [make_random_image_b64() for _ in range(8)],
        "model_precision": "fp32",
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["batch_size"] == 8
    assert len(data["results"]) == 8


def test_batch_predict_too_large(client):
    payload = {
        "images_b64":      [make_random_image_b64() for _ in range(9)],
        "model_precision": "fp32",
    }
    resp = client.post("/predict/batch", json=payload)
    # Either 422 (pydantic max_length) or 400 (our custom check)
    assert resp.status_code in (400, 422)


# ── Tests: /metrics ────────────────────────────────────────────────────────────

def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "requests_total" in data
    assert "errors_total"   in data
    assert "avg_inference_ms" in data
