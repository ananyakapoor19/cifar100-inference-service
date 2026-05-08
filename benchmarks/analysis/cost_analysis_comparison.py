"""
cost_analysis_comparison.py – Cross-platform cost analysis: Cloud Run CPU vs GCE GPU.

Pricing (us-central1, as of 2024):
  Cloud Run (always-allocated CPU):
    CPU:      $0.00002400 per vCPU-second  × 2 vCPU
    Memory:   $0.00000250 per GiB-second   × 2 GiB
    Requests: $0.40 per million

  GCE n1-standard-4 + NVIDIA T4 (on-demand):
    n1-standard-4:  $0.190004/hour  (4 vCPU, 15 GB RAM)
    NVIDIA T4 GPU:  $0.350000/hour
    Total:          $0.540004/hour  → $0.00015001/second (VM always running)

Usage:
    python benchmarks/analysis/cost_analysis_comparison.py \\
        --cpu-summary  results/cloud_run_cpu/charts/summary.json \\
        --gpu-summary  results/cloud_run_gpu_fp32/summary.json \\
        --output-dir   results/cost_comparison
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Pricing constants ──────────────────────────────────────────────────────────
# Cloud Run (2 vCPU, 2 GiB, always-allocated)
CR_CPU_PRICE_PER_VCPU_S = 0.00002400
CR_MEM_PRICE_PER_GIB_S  = 0.00000250
CR_REQUEST_PRICE         = 0.40 / 1e6
CR_CPU_COUNT             = 2
CR_MEMORY_GIB            = 2.0

# GCE n1-standard-4 + T4 (on-demand, us-central1)
GCE_N1_STANDARD4_HOURLY  = 0.190004   # 4 vCPU, 15 GB RAM
GCE_T4_GPU_HOURLY        = 0.350000   # NVIDIA T4
GCE_TOTAL_HOURLY         = GCE_N1_STANDARD4_HOURLY + GCE_T4_GPU_HOURLY
GCE_COST_PER_SECOND      = GCE_TOTAL_HOURLY / 3600.0

CONCURRENCIES = [10, 50, 200]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
})


# ── Cost calculation ───────────────────────────────────────────────────────────

def compute_cpu_costs(df: pd.DataFrame) -> pd.DataFrame:
    avg_s = df["avg_ms"] / 1000.0
    df = df.copy()
    df["platform"] = "Cloud Run CPU"
    df["cost_per_request"] = (
        avg_s * CR_CPU_COUNT  * CR_CPU_PRICE_PER_VCPU_S
        + avg_s * CR_MEMORY_GIB * CR_MEM_PRICE_PER_GIB_S
        + CR_REQUEST_PRICE
    )
    df["cost_per_image"]    = df["cost_per_request"] / df["batch_size"]
    df["cost_per_1k_images"] = df["cost_per_image"] * 1000
    df["images_per_dollar"] = 1.0 / df["cost_per_image"]
    df["hourly_cost_usd"]   = df["cost_per_request"] * df["rps"] * 3600
    return df


def compute_gpu_costs(df: pd.DataFrame) -> pd.DataFrame:
    # GCE VM always running: cost = latency × per-second rate
    avg_s = df["avg_ms"] / 1000.0
    df = df.copy()
    df["platform"] = "GCE GPU (T4)"
    df["cost_per_request"] = avg_s * GCE_COST_PER_SECOND
    df["cost_per_image"]    = df["cost_per_request"] / df["batch_size"]
    df["cost_per_1k_images"] = df["cost_per_image"] * 1000
    df["images_per_dollar"] = 1.0 / df["cost_per_image"]
    df["hourly_cost_usd"]   = GCE_TOTAL_HOURLY   # flat: VM runs regardless of load
    return df


def load(path: str):
    with open(path) as f:
        return json.load(f)


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Charts ─────────────────────────────────────────────────────────────────────

def plot_cost_per_image_comparison(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame,
                                    output_dir: str) -> None:
    """Grouped bar: cost-per-image (µ$) — CPU FP32/INT8 vs GPU FP32, batch=1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(CONCURRENCIES))
    w = 0.22
    configs = [
        (cpu_df, "fp32", 1, "#4C72B0", "CPU FP32 b=1"),
        (cpu_df, "int8", 1, "#DD8452", "CPU INT8 b=1"),
        (gpu_df, "fp32", 1, "#2ca02c", "GPU FP32 b=1"),
        (cpu_df, "fp32", 8, "#9467bd", "CPU FP32 b=8"),
    ]

    for idx, (df, prec, bs, color, label) in enumerate(configs):
        vals = []
        for c in CONCURRENCIES:
            row = df[(df["precision"] == prec) & (df["batch_size"] == bs)
                     & (df["concurrency"] == c)]
            vals.append(row["cost_per_image"].values[0] * 1e6 if not row.empty else 0)
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                        f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in CONCURRENCIES])
    ax.set_xlabel("Concurrency (users)")
    ax.set_ylabel("Cost per image (µ$)")
    ax.set_title("Cost per Image: Cloud Run CPU vs GCE GPU", fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, os.path.join(output_dir, "cost_cpu_vs_gpu.png"))


def plot_images_per_dollar(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame,
                            output_dir: str) -> None:
    """Images processed per dollar — higher is better."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONCURRENCIES))
    w = 0.22
    configs = [
        (cpu_df, "fp32", 1, "#4C72B0", "CPU FP32 b=1"),
        (cpu_df, "int8", 1, "#DD8452", "CPU INT8 b=1"),
        (gpu_df, "fp32", 1, "#2ca02c", "GPU FP32 b=1"),
        (cpu_df, "fp32", 8, "#9467bd", "CPU FP32 b=8"),
    ]

    for idx, (df, prec, bs, color, label) in enumerate(configs):
        vals = []
        for c in CONCURRENCIES:
            row = df[(df["precision"] == prec) & (df["batch_size"] == bs)
                     & (df["concurrency"] == c)]
            vals.append(row["images_per_dollar"].values[0] / 1000 if not row.empty else 0)
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                        f"{val:.0f}k", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in CONCURRENCIES])
    ax.set_xlabel("Concurrency (users)")
    ax.set_ylabel("Images per dollar (thousands)")
    ax.set_title("Cost Efficiency: Images per Dollar (higher = better)", fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, os.path.join(output_dir, "images_per_dollar_comparison.png"))


def plot_latency_vs_cost(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame,
                          output_dir: str) -> None:
    """Scatter: p50 latency vs cost-per-image for all configs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Latency vs Cost Trade-off (batch=1, all concurrencies)",
                 fontweight="bold")

    scatter_configs = [
        (cpu_df, "fp32", 1, "#4C72B0", "CPU FP32 b=1"),
        (cpu_df, "int8", 1, "#DD8452", "CPU INT8 b=1"),
        (gpu_df, "fp32", 1, "#2ca02c", "GPU FP32 b=1"),
    ]
    for df, prec, bs, color, label in scatter_configs:
        sub = df[(df["precision"] == prec) & (df["batch_size"] == bs)]
        if sub.empty:
            continue
        ax.scatter(sub["p50_ms"], sub["cost_per_image"] * 1e6,
                   color=color, label=label, s=100, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(f"c={int(row['concurrency'])}",
                        (row["p50_ms"], row["cost_per_image"] * 1e6),
                        textcoords="offset points", xytext=(6, 3), fontsize=8)

    ax.set_xlabel("p50 Latency (ms) — lower is better →")
    ax.set_ylabel("Cost per image (µ$) — lower is better ↓")
    ax.legend(fontsize=9)
    # Annotate ideal quadrant
    ax.text(0.05, 0.95, "← Ideal (fast + cheap)", transform=ax.transAxes,
            fontsize=9, color="green", va="top")
    save(fig, os.path.join(output_dir, "latency_vs_cost.png"))


def plot_throughput_comparison(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame,
                                output_dir: str) -> None:
    """Side-by-side bars: RPS (batch=1) CPU vs GPU."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RPS
    ax = axes[0]
    x = np.arange(len(CONCURRENCIES))
    w = 0.25
    configs = [
        (cpu_df, "fp32", "#4C72B0", "CPU FP32"),
        (cpu_df, "int8", "#DD8452", "CPU INT8"),
        (gpu_df, "fp32", "#2ca02c", "GPU FP32"),
    ]
    for idx, (df, prec, color, label) in enumerate(configs):
        vals = [
            df[(df["precision"] == prec) & (df["batch_size"] == 1)
               & (df["concurrency"] == c)]["rps"].values
            for c in CONCURRENCIES
        ]
        vals = [v[0] if len(v) else 0 for v in vals]
        bars = ax.bar(x + (idx - 1) * w, vals, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                        f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in CONCURRENCIES])
    ax.set_ylabel("Requests / second")
    ax.set_title("Throughput (batch=1)", fontweight="bold")
    ax.legend(fontsize=9)

    # Right: p50 latency
    ax2 = axes[1]
    for idx, (df, prec, color, label) in enumerate(configs):
        vals = [
            df[(df["precision"] == prec) & (df["batch_size"] == 1)
               & (df["concurrency"] == c)]["p50_ms"].values
            for c in CONCURRENCIES
        ]
        vals = [v[0] if len(v) else 0 for v in vals]
        ax2.bar(x + (idx - 1) * w, vals, w, label=label, color=color, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"c={c}" for c in CONCURRENCIES])
    ax2.set_ylabel("p50 Latency (ms)")
    ax2.set_title("p50 Latency (batch=1)", fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Cloud Run CPU vs GCE GPU — Throughput & Latency",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "throughput_latency_comparison.png"))


# ── Console output ─────────────────────────────────────────────────────────────

def print_pricing_assumptions():
    print("\n" + "=" * 70)
    print("PRICING ASSUMPTIONS")
    print("=" * 70)
    print(f"\nCloud Run CPU (us-central1, always-allocated):")
    print(f"  CPU:      ${CR_CPU_PRICE_PER_VCPU_S:.8f}/vCPU-s × {CR_CPU_COUNT} vCPU")
    print(f"  Memory:   ${CR_MEM_PRICE_PER_GIB_S:.8f}/GiB-s  × {CR_MEMORY_GIB} GiB")
    print(f"  Requests: ${CR_REQUEST_PRICE:.7f}/request")
    print(f"  Cost model: per-request duration (pay only when serving)")

    print(f"\nGCE GPU (us-central1, on-demand):")
    print(f"  n1-standard-4:  ${GCE_N1_STANDARD4_HOURLY:.6f}/hour (4 vCPU, 15 GB)")
    print(f"  NVIDIA T4 GPU:  ${GCE_T4_GPU_HOURLY:.6f}/hour")
    print(f"  Total:          ${GCE_TOTAL_HOURLY:.6f}/hour = ${GCE_COST_PER_SECOND:.8f}/second")
    print(f"  Cost model: flat hourly rate (VM always running)")


def print_detailed_table(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame) -> None:
    combined = pd.concat([cpu_df, gpu_df], ignore_index=True)

    print("\n" + "=" * 70)
    print("DETAILED COST BREAKDOWN")
    print("=" * 70)

    cols = ["platform", "precision", "batch_size", "concurrency",
            "rps", "avg_ms", "p50_ms", "failures",
            "cost_per_request", "cost_per_image", "cost_per_1k_images",
            "images_per_dollar", "hourly_cost_usd"]

    out = combined[cols].copy()
    out["cost_per_request"]   = out["cost_per_request"].map("${:.6f}".format)
    out["cost_per_image"]     = out["cost_per_image"].map("${:.6f}".format)
    out["cost_per_1k_images"] = out["cost_per_1k_images"].map("${:.4f}".format)
    out["images_per_dollar"]  = out["images_per_dollar"].map("{:,.0f}".format)
    out["hourly_cost_usd"]    = out["hourly_cost_usd"].map("${:.4f}".format)
    out["avg_ms"]             = out["avg_ms"].map("{:.0f}ms".format)
    out["p50_ms"]             = out["p50_ms"].map("{:.0f}ms".format)
    out["rps"]                = out["rps"].map("{:.1f}".format)
    out["failures"]           = out["failures"].map("{:.0f}".format)

    print(out.to_string(index=False))


def print_comparison_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: GPU vs CPU (batch=1, FP32, no failures)")
    print("=" * 70)

    for c in CONCURRENCIES:
        cpu_row = cpu_df[(cpu_df["precision"] == "fp32") & (cpu_df["batch_size"] == 1)
                         & (cpu_df["concurrency"] == c) & (cpu_df["failures"] == 0)]
        gpu_row = gpu_df[(gpu_df["precision"] == "fp32") & (gpu_df["batch_size"] == 1)
                         & (gpu_df["concurrency"] == c)]
        if cpu_row.empty or gpu_row.empty:
            continue

        cpu_r = cpu_row.iloc[0]
        gpu_r = gpu_row.iloc[0]

        latency_speedup = cpu_r["p50_ms"] / gpu_r["p50_ms"]
        rps_ratio       = gpu_r["rps"] / cpu_r["rps"]
        cost_ratio      = gpu_r["cost_per_image"] / cpu_r["cost_per_image"]

        print(f"\n  Concurrency c={c}:")
        print(f"    Latency   — CPU p50: {cpu_r['p50_ms']:.0f}ms  |  GPU p50: {gpu_r['p50_ms']:.0f}ms"
              f"  → GPU is {latency_speedup:.1f}× faster")
        print(f"    Throughput— CPU RPS: {cpu_r['rps']:.1f}  |  GPU RPS: {gpu_r['rps']:.1f}"
              f"  → GPU is {rps_ratio:.1f}× higher")
        print(f"    Cost/img  — CPU: ${cpu_r['cost_per_image']:.6f}  |  GPU: ${gpu_r['cost_per_image']:.6f}"
              f"  → GPU is {cost_ratio:.1f}× {'more expensive' if cost_ratio > 1 else 'cheaper'}")
        print(f"    CPU images/$: {cpu_r['images_per_dollar']:,.0f}  |  GPU images/$: {gpu_r['images_per_dollar']:,.0f}")

    # INT8 vs GPU comparison
    print("\n" + "-" * 70)
    print("  INT8 CPU vs GPU FP32 (batch=1, c=10, optimal configs):")
    int8_best = cpu_df[(cpu_df["precision"] == "int8") & (cpu_df["batch_size"] == 1)
                       & (cpu_df["concurrency"] == 10)]
    gpu_best  = gpu_df[(gpu_df["precision"] == "fp32") & (gpu_df["batch_size"] == 1)
                       & (gpu_df["concurrency"] == 10)]
    if not int8_best.empty and not gpu_best.empty:
        i8 = int8_best.iloc[0]
        gp = gpu_best.iloc[0]
        print(f"    INT8 CPU — p50: {i8['p50_ms']:.0f}ms, ${i8['cost_per_image']:.6f}/img,"
              f" {i8['images_per_dollar']:,.0f} images/$")
        print(f"    GPU FP32 — p50: {gp['p50_ms']:.0f}ms, ${gp['cost_per_image']:.6f}/img,"
              f" {gp['images_per_dollar']:,.0f} images/$")
        cost_ratio = gp["cost_per_image"] / i8["cost_per_image"]
        lat_ratio  = i8["p50_ms"] / gp["p50_ms"]
        print(f"    GPU is {lat_ratio:.1f}× faster in latency but {cost_ratio:.1f}× "
              f"{'more expensive' if cost_ratio > 1 else 'cheaper'} per image")

    # Best overall
    print("\n" + "-" * 70)
    print("  Best configurations overall (no failures):")
    cpu_valid = cpu_df[cpu_df["failures"] == 0]
    gpu_valid = gpu_df[gpu_df["failures"] == 0]
    all_valid = pd.concat([cpu_valid, gpu_valid])

    all_valid = all_valid.reset_index(drop=True)
    cheapest    = all_valid.loc[all_valid["cost_per_image"].idxmin()].to_dict()
    fastest     = all_valid.loc[all_valid["p50_ms"].idxmin()].to_dict()
    highest_rps = all_valid.loc[(all_valid["rps"] * all_valid["batch_size"]).idxmax()].to_dict()

    print(f"    Cheapest:     {cheapest['platform']} | {str(cheapest['precision']).upper()} "
          f"b={int(cheapest['batch_size'])} c={int(cheapest['concurrency'])} → "
          f"${float(cheapest['cost_per_image']):.6f}/image ({float(cheapest['images_per_dollar']):,.0f} images/$)")
    print(f"    Fastest p50:  {fastest['platform']} | {str(fastest['precision']).upper()} "
          f"b={int(fastest['batch_size'])} c={int(fastest['concurrency'])} → "
          f"{float(fastest['p50_ms']):.0f}ms p50")
    print(f"    Highest RPS:  {highest_rps['platform']} | {str(highest_rps['precision']).upper()} "
          f"b={int(highest_rps['batch_size'])} c={int(highest_rps['concurrency'])} → "
          f"{float(highest_rps['rps']) * int(highest_rps['batch_size']):.1f} images/s")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-summary",  default="results/cloud_run_cpu/charts/summary.json")
    parser.add_argument("--gpu-summary",  default="results/cloud_run_gpu_fp32/summary.json")
    parser.add_argument("--output-dir",   default="results/cost_comparison")
    args = parser.parse_args()

    print_pricing_assumptions()

    cpu_data = load(args.cpu_summary)
    gpu_data = load(args.gpu_summary)

    cpu_df = compute_cpu_costs(pd.DataFrame(cpu_data))
    gpu_df = compute_gpu_costs(pd.DataFrame(gpu_data))

    print_detailed_table(cpu_df, gpu_df)
    print_comparison_summary(cpu_df, gpu_df)

    print("\nGenerating charts…")
    os.makedirs(args.output_dir, exist_ok=True)
    plot_cost_per_image_comparison(cpu_df, gpu_df, args.output_dir)
    plot_images_per_dollar(cpu_df, gpu_df, args.output_dir)
    plot_latency_vs_cost(cpu_df, gpu_df, args.output_dir)
    plot_throughput_comparison(cpu_df, gpu_df, args.output_dir)

    # Save enriched combined data
    combined = pd.concat([cpu_df, gpu_df], ignore_index=True)
    out_path = os.path.join(args.output_dir, "cost_comparison.json")
    combined.to_json(out_path, orient="records", indent=2)
    print(f"  Saved → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
