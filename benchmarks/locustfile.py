"""
locustfile.py – Locust load test for the CIFAR-100 inference service.

Covers the full experiment matrix:
    precision  ∈ {fp32, int8}
    batch_size ∈ {1, 8}
    concurrency levels are controlled externally via --users flag

Run headless example (single cell):
    locust -f benchmarks/locustfile.py \
        --headless \
        --users 50 --spawn-rate 5 \
        --run-time 60s \
        --host http://localhost:8080 \
        --csv results/fp32_batch1_c50

Environment variables (override at runtime):
    PRECISION   – fp32 | int8  (default: fp32)
    BATCH_SIZE  – 1 | 8        (default: 1)
    RETURN_PROF – true | false (default: false)
"""

import base64
import io
import json
import os
import random

import numpy as np
from locust import HttpUser, between, events, task
from PIL import Image


# ── Synthetic test image factory ───────────────────────────────────────────────

def _make_random_jpeg(width: int = 64, height: int = 64) -> str:
    """Generate a random RGB JPEG and return as base64 string."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Pre-generate a pool of 100 synthetic images to avoid per-request overhead
_IMAGE_POOL = [_make_random_jpeg() for _ in range(100)]


def _random_image() -> str:
    return random.choice(_IMAGE_POOL)


# ── Read experiment config from env ───────────────────────────────────────────
PRECISION    = os.environ.get("PRECISION",   "fp32")
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "1"))
RETURN_PROF  = os.environ.get("RETURN_PROF", "false").lower() == "true"


# ── User classes ───────────────────────────────────────────────────────────────

class SingleImageUser(HttpUser):
    """
    Fires POST /predict with a single image.
    Use when BATCH_SIZE=1.
    """
    wait_time = between(0.01, 0.05)   # tight loop for throughput testing
    weight    = 1 if BATCH_SIZE == 1 else 0

    @task
    def predict_single(self):
        payload = {
            "image_b64":       _random_image(),
            "model_precision": PRECISION,
            "return_profile":  RETURN_PROF,
        }
        with self.client.post(
            "/predict",
            json=payload,
            name=f"/predict [{PRECISION}|batch=1]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("predictions"):
                    resp.failure("Empty predictions list")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(weight=0)
    def health_check(self):
        self.client.get("/health", name="/health")


class BatchImageUser(HttpUser):
    """
    Fires POST /predict/batch with BATCH_SIZE images.
    Use when BATCH_SIZE=8.
    """
    wait_time = between(0.05, 0.1)
    weight    = 0 if BATCH_SIZE == 1 else 1

    @task
    def predict_batch(self):
        payload = {
            "images_b64":      [_random_image() for _ in range(BATCH_SIZE)],
            "model_precision": PRECISION,
            "return_profile":  RETURN_PROF,
        }
        with self.client.post(
            "/predict/batch",
            json=payload,
            name=f"/predict/batch [{PRECISION}|batch={BATCH_SIZE}]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("batch_size") != BATCH_SIZE:
                    resp.failure(f"Expected batch_size={BATCH_SIZE}, got {data.get('batch_size')}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")


# ── Custom CSV result listener ─────────────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kw):
    """Print a summary row to stdout for easy log scraping."""
    stats = environment.runner.stats
    total = stats.total
    if total.num_requests == 0:
        return

    summary = {
        "precision":       PRECISION,
        "batch_size":      BATCH_SIZE,
        "num_users":       environment.runner.target_user_count,
        "requests":        total.num_requests,
        "failures":        total.num_failures,
        "rps":             round(total.current_rps, 2),
        "p50_ms":          round(total.get_response_time_percentile(0.50), 2),
        "p95_ms":          round(total.get_response_time_percentile(0.95), 2),
        "p99_ms":          round(total.get_response_time_percentile(0.99), 2),
        "avg_ms":          round(total.avg_response_time, 2),
    }
    print("\n[SUMMARY]", json.dumps(summary))
