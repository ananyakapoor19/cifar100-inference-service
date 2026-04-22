"""
test_model.py – Unit tests for model-related utilities.

Run with:
    pytest tests/test_model.py -v
"""

import io
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from api.inference import (
    CIFAR100_CLASSES,
    decode_image,
    preprocess,
    preprocess_batch,
    run_inference,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_pil(width: int = 64, height: int = 64) -> Image.Image:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    make_pil(width, height).save(buf, format="JPEG")
    return buf.getvalue()


def tiny_model(num_classes: int = 100) -> nn.Module:
    """Lightweight model for fast tests (no GPU needed)."""
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.0),
        nn.Linear(in_features, num_classes),
    )
    m.eval()
    return m


# ── CIFAR-100 class list ───────────────────────────────────────────────────────

def test_cifar100_classes_count():
    assert len(CIFAR100_CLASSES) == 100


def test_cifar100_classes_unique():
    assert len(set(CIFAR100_CLASSES)) == 100


# ── Image decoding ─────────────────────────────────────────────────────────────

def test_decode_image_jpeg():
    img = decode_image(make_jpeg_bytes())
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_decode_image_invalid():
    with pytest.raises(Exception):
        decode_image(b"not an image")


# ── Preprocessing ──────────────────────────────────────────────────────────────

def test_preprocess_shape():
    tensor = preprocess(make_pil())
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_dtype():
    tensor = preprocess(make_pil())
    assert tensor.dtype == torch.float32


def test_preprocess_batch_shape():
    images = [make_pil() for _ in range(4)]
    tensor = preprocess_batch(images)
    assert tensor.shape == (4, 3, 224, 224)


def test_preprocess_normalisation():
    """Tensor values should be roughly in [-3, 3] after normalisation."""
    tensor = preprocess(make_pil())
    assert tensor.min().item() > -5.0
    assert tensor.max().item() <  5.0


# ── Inference ──────────────────────────────────────────────────────────────────

def test_run_inference_single():
    model  = tiny_model()
    tensor = preprocess(make_pil())
    device = torch.device("cpu")
    preds, latency_ms = run_inference(model, tensor, device, top_k=5)
    assert len(preds) == 1        # one image
    assert len(preds[0]) == 5     # top-5
    assert latency_ms > 0.0


def test_run_inference_batch():
    model  = tiny_model()
    images = [make_pil() for _ in range(3)]
    tensor = preprocess_batch(images)
    device = torch.device("cpu")
    preds, _ = run_inference(model, tensor, device, top_k=5)
    assert len(preds) == 3


def test_run_inference_prediction_format():
    model  = tiny_model()
    tensor = preprocess(make_pil())
    device = torch.device("cpu")
    preds, _ = run_inference(model, tensor, device, top_k=5)
    for p in preds[0]:
        assert "class_id"   in p
        assert "class_name" in p
        assert "confidence" in p
        assert 0 <= p["class_id"] < 100
        assert p["class_name"] in CIFAR100_CLASSES
        assert 0.0 <= p["confidence"] <= 1.0


def test_run_inference_top1_highest_confidence():
    model  = tiny_model()
    tensor = preprocess(make_pil())
    device = torch.device("cpu")
    preds, _ = run_inference(model, tensor, device, top_k=5)
    confs = [p["confidence"] for p in preds[0]]
    assert confs[0] == max(confs), "Top-1 should have highest confidence"


def test_run_inference_confidences_sum():
    """Top-5 confidences should be ≤ 1.0 (they're a subset of softmax)."""
    model  = tiny_model()
    tensor = preprocess(make_pil())
    device = torch.device("cpu")
    preds, _ = run_inference(model, tensor, device, top_k=5)
    conf_sum = sum(p["confidence"] for p in preds[0])
    assert conf_sum <= 1.0 + 1e-5
