"""
compare_results.py – Generate comparison charts across deployments.

Charts produced:
  1. fp32_cpu_vs_int8_cpu.png  – FP32 CPU vs INT8 CPU latency (RQ1 corrected)
  2. fp32_gpu_vs_fp32_cpu.png  – FP32 GPU vs FP32 CPU latency (partial RQ2)

Usage:
    python benchmarks/analysis/compare_results.py \
        --cpu-summary  results/cloud_run_cpu/charts/summary.json \
        --gpu-summary  results/cloud_run_gpu_fp32/summary.json \
        --output-dir   results/cloud_run_cpu/charts
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

CONCURRENCY = [10, 50, 200]

def load(path):
    with open(path) as f:
        return json.load(f)

def lookup(data, precision, batch_size, concurrency, key):
    for r in data:
        if (r["precision"] == precision
                and r["batch_size"] == batch_size
                and r["concurrency"] == concurrency):
            return r[key]
    return None

# ── Chart 1: FP32 CPU vs INT8 CPU ─────────────────────────────────────────────
def plot_fp32_vs_int8_cpu(cpu_data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: batch=1 latency
    ax = axes[0]
    fp32_p50 = [lookup(cpu_data, "fp32", 1, c, "p50_ms") for c in CONCURRENCY]
    int8_p50 = [lookup(cpu_data, "int8", 1, c, "p50_ms") for c in CONCURRENCY]
    fp32_p95 = [lookup(cpu_data, "fp32", 1, c, "p95_ms") for c in CONCURRENCY]
    int8_p95 = [lookup(cpu_data, "int8", 1, c, "p95_ms") for c in CONCURRENCY]

    x = np.arange(len(CONCURRENCY))
    ax.plot(x, fp32_p50, "o-",  color="#4C72B0", lw=2,   label="FP32 p50")
    ax.plot(x, fp32_p95, "s--", color="#4C72B0", lw=1.5, label="FP32 p95", alpha=0.7)
    ax.plot(x, int8_p50, "o-",  color="#DD8452", lw=2,   label="INT8 p50")
    ax.plot(x, int8_p95, "s--", color="#DD8452", lw=1.5, label="INT8 p95", alpha=0.7)

    # Speedup annotations
    for i, c in enumerate(CONCURRENCY):
        if fp32_p50[i] and int8_p50[i]:
            speedup = fp32_p50[i] / int8_p50[i]
            ax.annotate(f"{speedup:.1f}×\nfaster",
                        xy=(x[i], int8_p50[i]),
                        xytext=(x[i] + 0.08, int8_p50[i] * 0.7),
                        fontsize=9, color="green", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in CONCURRENCY])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FP32 vs INT8 — batch=1 (CPU)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: batch=1 RPS
    ax2 = axes[1]
    fp32_rps = [lookup(cpu_data, "fp32", 1, c, "rps") for c in CONCURRENCY]
    int8_rps = [lookup(cpu_data, "int8", 1, c, "rps") for c in CONCURRENCY]

    w = 0.3
    bars1 = ax2.bar(x - w/2, fp32_rps, w, label="FP32", color="#4C72B0", alpha=0.85)
    bars2 = ax2.bar(x + w/2, int8_rps, w, label="INT8", color="#DD8452", alpha=0.85)

    for bar, val in zip(list(bars1) + list(bars2), fp32_rps + int8_rps):
        if val:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                     f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"c={c}" for c in CONCURRENCY])
    ax2.set_ylabel("Requests / second")
    ax2.set_title("FP32 vs INT8 — Throughput batch=1 (CPU)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "RQ1 · FP32 CPU vs INT8 CPU — INT8 is ~1.7× faster on CPU",
        fontweight="bold", fontsize=13, y=1.02
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "fp32_cpu_vs_int8_cpu.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Chart 2: FP32 GPU vs FP32 CPU ─────────────────────────────────────────────
def plot_fp32_gpu_vs_cpu(gpu_data, cpu_data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: batch=1 latency
    ax = axes[0]
    gpu_p50 = [lookup(gpu_data, "fp32", 1, c, "p50_ms") for c in CONCURRENCY]
    cpu_p50 = [lookup(cpu_data, "fp32", 1, c, "p50_ms") for c in CONCURRENCY]
    gpu_p95 = [lookup(gpu_data, "fp32", 1, c, "p95_ms") for c in CONCURRENCY]
    cpu_p95 = [lookup(cpu_data, "fp32", 1, c, "p95_ms") for c in CONCURRENCY]

    x = np.arange(len(CONCURRENCY))
    ax.plot(x, gpu_p50, "o-",  color="#2ca02c", lw=2,   label="GPU p50")
    ax.plot(x, gpu_p95, "s--", color="#2ca02c", lw=1.5, label="GPU p95", alpha=0.7)
    ax.plot(x, cpu_p50, "o-",  color="#4C72B0", lw=2,   label="CPU p50")
    ax.plot(x, cpu_p95, "s--", color="#4C72B0", lw=1.5, label="CPU p95", alpha=0.7)

    # Speedup annotations
    for i, c in enumerate(CONCURRENCY):
        if gpu_p50[i] and cpu_p50[i]:
            speedup = cpu_p50[i] / gpu_p50[i]
            ax.annotate(f"GPU\n{speedup:.1f}× faster",
                        xy=(x[i], gpu_p50[i]),
                        xytext=(x[i] + 0.08, gpu_p50[i] * 1.4),
                        fontsize=9, color="green", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in CONCURRENCY])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FP32 GPU vs FP32 CPU — batch=1", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: throughput
    ax2 = axes[1]
    gpu_rps = [lookup(gpu_data, "fp32", 1, c, "rps") for c in CONCURRENCY]
    cpu_rps = [lookup(cpu_data, "fp32", 1, c, "rps") for c in CONCURRENCY]

    w = 0.3
    bars1 = ax2.bar(x - w/2, gpu_rps, w, label="GPU (FP32)", color="#2ca02c", alpha=0.85)
    bars2 = ax2.bar(x + w/2, cpu_rps, w, label="CPU (FP32)", color="#4C72B0", alpha=0.85)

    for bar, val in zip(list(bars1) + list(bars2), gpu_rps + cpu_rps):
        if val:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                     f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"c={c}" for c in CONCURRENCY])
    ax2.set_ylabel("Requests / second")
    ax2.set_title("FP32 GPU vs FP32 CPU — Throughput batch=1", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "RQ2 · FP32 GPU vs FP32 CPU — GPU is 2–2.7× faster",
        fontweight="bold", fontsize=13, y=1.02
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "fp32_gpu_vs_fp32_cpu.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-summary",  required=True)
    parser.add_argument("--gpu-summary",  required=True)
    parser.add_argument("--output-dir",   required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cpu_data = load(args.cpu_summary)
    gpu_data = load(args.gpu_summary)

    print("Generating comparison charts…")
    plot_fp32_vs_int8_cpu(cpu_data, args.output_dir)
    plot_fp32_gpu_vs_cpu(gpu_data, cpu_data, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
