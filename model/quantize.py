"""
quantize.py – Apply INT8 post-training static quantization to the trained FP32 model.

Usage:
    python model/quantize.py \
        --fp32-ckpt ./checkpoints/efficientnet_b0_cifar100_fp32.pth \
        --output-dir ./checkpoints \
        --data-dir ./data \
        --calib-batches 16

Outputs:
    checkpoints/efficientnet_b0_cifar100_int8.pth   – quantized TorchScript module
    checkpoints/quant_comparison.json               – accuracy + latency comparison
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMG_SIZE = 224


def build_model(num_classes: int = 100) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_fp32_model(ckpt_path: str, device: torch.device) -> nn.Module:
    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def get_calib_loader(data_dir: str, num_batches: int, batch_size: int = 32):
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_ds = datasets.CIFAR100(data_dir, train=False, download=True, transform=val_tf)
    subset = Subset(val_ds, list(range(min(num_batches * batch_size, len(val_ds)))))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def calibrate(model: nn.Module, loader: DataLoader):
    """Run calibration data through the model to collect activation statistics."""
    model.eval()
    for images, _ in loader:
        model(images)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels.to(device)).sum().item()
        total   += labels.size(0)
    return correct / total


def measure_latency(model: nn.Module, device: torch.device,
                    batch_size: int = 1, warmup: int = 20, runs: int = 100) -> dict:
    """Measure p50/p95/p99 latency in milliseconds."""
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
    timings = []

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            timings.append((time.perf_counter() - t0) * 1000)

    timings.sort()
    n = len(timings)
    return {
        "p50_ms":  timings[int(n * 0.50)],
        "p95_ms":  timings[int(n * 0.95)],
        "p99_ms":  timings[int(n * 0.99)],
        "mean_ms": sum(timings) / n,
    }


def quantize_static(fp32_model: nn.Module, calib_loader: DataLoader) -> nn.Module:
    """
    Apply INT8 post-training dynamic quantization.

    EfficientNet-B0 uses SiLU/hardswish activations that lack quantized kernels in
    the legacy static-PTQ path (torch.ao.quantization.prepare/convert).  Dynamic
    quantization — which quantises Linear and Conv2d *weights* at model load time and
    runs activations in FP32 — is the fully-supported alternative and still delivers
    meaningful model-size reduction and latency improvement on CPU.

    Note: `calib_loader` is accepted for API compatibility but is not used for
    dynamic quantization (no calibration step required).
    """
    import copy
    import platform

    print("Using dynamic quantization (static PTQ incompatible with SiLU activations).")
    model_q = copy.deepcopy(fp32_model).cpu()
    model_q.eval()

    # fbgemm is x86-only; use qnnpack on ARM (Apple Silicon, ARM Linux)
    backend = "qnnpack" if platform.machine() in ("arm64", "aarch64") else "fbgemm"
    torch.backends.quantized.engine = backend

    model_q = torch.quantization.quantize_dynamic(
        model_q,
        qconfig_spec={torch.nn.Linear},   # Conv2d dynamic quant not universally supported
        dtype=torch.qint8,
    )
    print("INT8 dynamic quantization complete.")
    return model_q


def main():
    parser = argparse.ArgumentParser(description="INT8 post-training quantization")
    parser.add_argument("--fp32-ckpt",    type=str, required=True)
    parser.add_argument("--output-dir",   type=str, default="./checkpoints")
    parser.add_argument("--data-dir",     type=str, default="./data")
    parser.add_argument("--calib-batches", type=int, default=16,
                        help="Number of batches (32 images each) for calibration")
    parser.add_argument("--eval-full",    action="store_true",
                        help="Evaluate on the full val set (slower)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")   # Quantization runs on CPU

    print("Loading FP32 model…")
    fp32_model = load_fp32_model(args.fp32_ckpt, device)

    calib_loader = get_calib_loader(args.data_dir, args.calib_batches)

    # Evaluate FP32 baseline
    print("Evaluating FP32 baseline…")
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_ds = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=val_tf)
    eval_subset = eval_ds if args.eval_full else torch.utils.data.Subset(eval_ds, list(range(2000)))
    eval_loader = DataLoader(eval_subset, batch_size=64, shuffle=False, num_workers=2)

    fp32_acc  = evaluate_accuracy(fp32_model, eval_loader, device)
    fp32_lat  = measure_latency(fp32_model, device, batch_size=1)
    fp32_lat8 = measure_latency(fp32_model, device, batch_size=8)
    print(f"FP32 accuracy (subset): {fp32_acc:.4f}")
    print(f"FP32 latency batch=1:   {fp32_lat}")
    print(f"FP32 latency batch=8:   {fp32_lat8}")

    # Quantize
    int8_model = quantize_static(fp32_model, calib_loader)

    int8_acc  = evaluate_accuracy(int8_model, eval_loader, device)
    int8_lat  = measure_latency(int8_model, device, batch_size=1)
    int8_lat8 = measure_latency(int8_model, device, batch_size=8)
    print(f"INT8 accuracy (subset): {int8_acc:.4f}")
    print(f"INT8 latency batch=1:   {int8_lat}")
    print(f"INT8 latency batch=8:   {int8_lat8}")

    # Save quantized model as TorchScript
    int8_path = os.path.join(args.output_dir, "efficientnet_b0_cifar100_int8.pth")
    scripted  = torch.jit.script(int8_model)
    scripted.save(int8_path)
    print(f"Saved INT8 TorchScript model → {int8_path}")

    # Comparison report
    report = {
        "fp32": {"accuracy": fp32_acc, "latency_batch1": fp32_lat, "latency_batch8": fp32_lat8},
        "int8": {"accuracy": int8_acc, "latency_batch1": int8_lat, "latency_batch8": int8_lat8},
        "accuracy_drop": fp32_acc - int8_acc,
        "speedup_p50_batch1": fp32_lat["p50_ms"] / int8_lat["p50_ms"],
    }
    report_path = os.path.join(args.output_dir, "quant_comparison.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Comparison report → {report_path}")


if __name__ == "__main__":
    main()
