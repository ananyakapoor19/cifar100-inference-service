# CIFAR-100 End-to-End Inference Service
### COMSE6998 – Applied Machine Learning in the Cloud

**Team:** Ananya Kapoor (ak5447) · Tanmay Agrawal (ta2832) · Madhav Rajkondawar (mr4650) · Tarandeep Singh (ts3747)

---

## Overview

An end-to-end image classification inference service built on EfficientNet-B0 fine-tuned on CIFAR-100 (100 classes). The service is deployed on GCP and supports both **FP32** and **INT8 quantized** model variants, served via a FastAPI REST API on Cloud Run (CPU) and a GCE VM with NVIDIA T4 (GPU).

The primary research questions are:
1. How does INT8 quantization affect accuracy vs. latency under real serving conditions?
2. How do CPU (Cloud Run) vs. GPU (GCE T4) instances impact throughput and cost?
3. At what concurrency levels do compute, memory, and network I/O become bottlenecks?

---

## Repository Structure

```
cifar100-inference-service/
├── model/
│   ├── train.py            # Fine-tune EfficientNet-B0 on CIFAR-100
│   ├── quantize.py         # INT8 post-training dynamic quantization
│   ├── evaluate.py         # Top-1/5 accuracy evaluation
│   └── upload_to_gcs.py    # Push checkpoints to GCS
├── api/
│   ├── main.py             # FastAPI app (POST /predict, POST /predict/batch, GET /health)
│   ├── inference.py        # ModelRegistry, preprocessing, forward pass
│   ├── monitoring.py       # GCP Cloud Monitoring custom metrics (background flush)
│   ├── profiling.py        # Per-request PyTorch Profiler integration
│   └── schemas.py          # Pydantic request/response models
├── docker/
│   ├── Dockerfile          # CPU image (Cloud Run)
│   ├── Dockerfile.gpu      # GPU image (GCE T4)
│   └── docker-compose.yml  # Local dev
├── benchmarks/
│   ├── locustfile.py       # Locust load test (full experiment matrix)
│   ├── run_experiments.sh  # Execute all matrix cells
│   └── analysis/
│       ├── plot_results.py # Parse CSVs → latency/throughput charts
│       └── cost_analysis.py# Cloud Run cost-per-image analysis + charts
├── infra/
│   ├── cloud_run/service.yaml   # Cloud Run Knative service spec
│   ├── gce/startup.sh           # GCE GPU VM bootstrap
│   └── gcs/setup.sh             # Bucket + IAM setup
├── tests/
│   ├── test_api.py         # FastAPI integration tests
│   └── test_model.py       # Model utility unit tests
├── deploy.sh               # Top-level build + deploy script
├── requirements.txt
└── requirements-dev.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Train the model

```bash
python model/train.py \
    --epochs 25 \
    --batch-size 128 \
    --lr 1e-3 \
    --output-dir ./checkpoints
```

Training uses a two-phase strategy:
- **Phase 1 (epochs 1–5):** head-only training (backbone frozen)
- **Phase 2 (epochs 6–25):** full fine-tuning with lower LR for the backbone

Expected val accuracy: ~75–78% Top-1 after 25 epochs on a single GPU.

### 3. Quantize to INT8

```bash
python model/quantize.py \
    --fp32-ckpt ./checkpoints/efficientnet_b0_cifar100_fp32.pth \
    --output-dir ./checkpoints \
    --data-dir ./data
```

This runs post-training dynamic quantization (Linear + Conv2d layers) and saves:
- `checkpoints/efficientnet_b0_cifar100_int8.pth` – TorchScript INT8 model
- `checkpoints/quant_comparison.json` – accuracy + latency comparison

### 4. Run the server locally

```bash
MODEL_FP32_PATH=./checkpoints/efficientnet_b0_cifar100_fp32.pth \
MODEL_INT8_PATH=./checkpoints/efficientnet_b0_cifar100_int8.pth \
python -m api.main
```

Cloud Monitoring is disabled automatically when `GOOGLE_CLOUD_PROJECT` is not
set, so local dev works without any GCP credentials. To enable it:

```bash
GOOGLE_CLOUD_PROJECT=your-project \
MODEL_FP32_PATH=./checkpoints/efficientnet_b0_cifar100_fp32.pth \
MODEL_INT8_PATH=./checkpoints/efficientnet_b0_cifar100_int8.pth \
python -m api.main
```

Custom metrics are flushed every 60 seconds (override with `CM_FLUSH_INTERVAL=N`).

Test with curl:
```bash
# Encode an image
B64=$(base64 -w 0 my_image.jpg)

# Single prediction
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d "{\"image_b64\": \"$B64\", \"model_precision\": \"fp32\"}"

# With profiling breakdown
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d "{\"image_b64\": \"$B64\", \"model_precision\": \"fp32\", \"return_profile\": true}"

# Batch (up to 8 images)
curl -X POST http://localhost:8080/predict/batch \
    -H "Content-Type: application/json" \
    -d "{\"images_b64\": [\"$B64\", \"$B64\"], \"model_precision\": \"int8\"}"
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## GCP Deployment

### One-time setup

```bash
export PROJECT_ID=your-gcp-project
export BUCKET=your-gcs-bucket

# Create bucket + service account + IAM
./infra/gcs/setup.sh

# Upload checkpoints
python model/upload_to_gcs.py \
    --bucket $BUCKET \
    --files checkpoints/efficientnet_b0_cifar100_fp32.pth \
            checkpoints/efficientnet_b0_cifar100_int8.pth
```

### Deploy to Cloud Run (CPU)

```bash
PROJECT_ID=$PROJECT_ID BUCKET=$BUCKET ./deploy.sh cpu
```

Auto-scaling is configured to 0–10 replicas, scaling up when per-instance concurrency exceeds 5.

### Deploy to GCE (GPU – NVIDIA T4)

```bash
# Build + push GPU image
PROJECT_ID=$PROJECT_ID BUCKET=$BUCKET ./deploy.sh gpu

# Create GCE instance and run startup script
gcloud compute instances create cifar100-gpu-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=common-cu118 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata-from-file=startup-script=infra/gce/startup.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

---

## Benchmarking

### Run the full experiment matrix

```bash
# Against Cloud Run
HOST=https://your-cloud-run-url ./benchmarks/run_experiments.sh

# Against local server
HOST=http://localhost:8080 ./benchmarks/run_experiments.sh
```

Matrix: `{fp32, int8} × {batch=1, batch=8} × {concurrency=10, 50, 200}` = **12 cells**

Each cell records p50/p95/p99 latency, RPS, and failure rate.

### Generate latency/throughput charts

```bash
python benchmarks/analysis/plot_results.py \
    --results-dir ./results \
    --output-dir  ./results/charts
```

Charts produced:
- `latency_throughput.png` – p50/p95/p99 vs. RPS
- `concurrency_heatmap.png` – p95 latency heatmap
- `batch_comparison.png` – batch=1 vs batch=8 RPS
- `speedup_quantization.png` – FP32 vs INT8 latency with speedup annotations

### Generate cost analysis

```bash
python benchmarks/analysis/cost_analysis.py \
    --summary  results/cloud_run_cpu/charts/summary.json \
    --output-dir results/cloud_run_cpu/charts
```

Uses Cloud Run always-allocated pricing (2 vCPU, 2 GiB) to compute cost per
request and cost per image for every benchmark configuration. Charts produced:
- `cost_per_image.png` – cost per image by config × concurrency
- `cost_vs_throughput.png` – cost-efficiency frontier (images/s vs cost/image)
- `cost_fp32_vs_int8.png` – FP32 vs INT8 cost comparison with ratio annotations
- `cost_analysis.json` – full enriched table with all cost metrics

---

## API Reference

| Method | Endpoint         | Description                              |
|--------|-----------------|------------------------------------------|
| GET    | `/health`        | Liveness check, loaded models, device    |
| POST   | `/predict`       | Single-image inference                   |
| POST   | `/predict/batch` | Batch inference (up to 8 images)         |
| GET    | `/metrics`       | Request counters + average latency       |

### Request schema (`/predict`)

```json
{
  "image_b64": "<base64-encoded JPEG/PNG>",
  "model_precision": "fp32",
  "return_profile": false
}
```

### Response schema

```json
{
  "predictions": [
    {"class_id": 42, "class_name": "leopard", "confidence": 0.87},
    ...
  ],
  "model_precision": "fp32",
  "inference_ms": 12.4,
  "profile": {
    "preprocess_ms": 2.1,
    "forward_ms": 8.9,
    "postprocess_ms": 1.4,
    "total_ms": 12.4
  }
}
```

---

## Task Distribution

| Member | Responsibilities |
|--------|-----------------|
| Ananya Kapoor (ak5447) | Model training, quantization, GCS upload |
| Tanmay Agrawal (ta2832) | FastAPI server, Docker, Cloud Run deployment |
| Madhav Rajkondawar (mr4650) | GCE GPU deployment, CUDA profiling |
| Tarandeep Singh (ts3747) | Locust benchmarking, analysis charts, report |

---

## Course Relevance

- **Cloud Computing:** GCP Cloud Run (serverless, 0–10 auto-scaling replicas), GCE T4 GPU instance, GCS model storage, Cloud Monitoring custom metrics (request latency, inference latency, error rate pushed as time-series from a background thread).
- **Deep Neural Networks:** EfficientNet-B0 transfer learning on CIFAR-100 with two-phase fine-tuning; INT8 post-training dynamic quantization (Linear + Conv2d) via `torch.quantization`; accuracy vs. latency trade-off analysis.
- **Performance Analysis:** Systematic Locust benchmarking across a 12-cell matrix (precision × batch × concurrency); PyTorch Profiler traces decomposing per-request latency into preprocess/forward/postprocess stages; latency-throughput curves, heatmaps, and Cloud Run cost-per-image analysis.
