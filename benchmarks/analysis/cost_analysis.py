"""
cost_analysis.py – Compute Cloud Run cost metrics from Locust benchmark results.

Cloud Run pricing (us-central1, always-allocated CPU, as of 2024):
    CPU:      $0.00002400 per vCPU-second
    Memory:   $0.00000250 per GB-second
    Requests: $0.40 per million  ($0.0000004 per request)

Service configuration (infra/cloud_run/service.yaml):
    CPU:    2 vCPU
    Memory: 2 GiB

Cost model per request:
    cost = (avg_latency_s × CPU_COUNT  × CPU_PRICE_PER_VCPU_S)
         + (avg_latency_s × MEMORY_GIB × MEM_PRICE_PER_GIB_S)
         + REQUEST_PRICE

    cost_per_image = cost_per_request / batch_size   (amortises batch overhead)

Usage:
    python benchmarks/analysis/cost_analysis.py \\
        --summary results/cloud_run_cpu/charts/summary.json \\
        --output-dir results/cloud_run_cpu/charts
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Cloud Run pricing constants ────────────────────────────────────────────────
CPU_PRICE_PER_VCPU_S  = 0.00002400   # $/vCPU-second (always-allocated)
MEM_PRICE_PER_GIB_S   = 0.00000250   # $/GiB-second  (always-allocated)
REQUEST_PRICE         = 0.40 / 1e6   # $/request     ($0.40 per million)

# Resources allocated per container replica (matches service.yaml)
CPU_COUNT   = 2      # vCPU
MEMORY_GIB  = 2.0    # GiB

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

COLORS = {
    "fp32_b1": "#2196F3",
    "fp32_b8": "#0D47A1",
    "int8_b1": "#FF5722",
    "int8_b8": "#BF360C",
}


# ── Cost calculation ───────────────────────────────────────────────────────────

def compute_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Add cost columns to the benchmark DataFrame."""
    avg_s = df["avg_ms"] / 1000.0

    df["cost_per_request"] = (
        avg_s * CPU_COUNT  * CPU_PRICE_PER_VCPU_S
        + avg_s * MEMORY_GIB * MEM_PRICE_PER_GIB_S
        + REQUEST_PRICE
    )
    # Per-image cost amortises the batch overhead across images in the batch.
    df["cost_per_image"]   = df["cost_per_request"] / df["batch_size"]

    # Dollar cost to process 1 000 images
    df["cost_per_1k_images"] = df["cost_per_image"] * 1000

    # Images processed per dollar (efficiency, higher = better)
    df["images_per_dollar"] = 1.0 / df["cost_per_image"]

    # Projected hourly cost when running at the observed RPS
    df["hourly_cost_usd"] = df["cost_per_request"] * df["rps"] * 3600

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Charts ─────────────────────────────────────────────────────────────────────

def plot_cost_per_image(df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart: cost-per-image for each config × concurrency."""
    concurrencies = sorted(df["concurrency"].unique())
    configs = [("fp32", 1), ("fp32", 8), ("int8", 1), ("int8", 8)]
    x = np.arange(len(concurrencies))
    w = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, (prec, bs) in enumerate(configs):
        key   = f"{prec}_b{bs}"
        color = COLORS.get(key, "#888")
        vals  = [
            df[(df["precision"] == prec) & (df["batch_size"] == bs)
               & (df["concurrency"] == c)]["cost_per_image"].values
            for c in concurrencies
        ]
        vals = [v[0] * 1e6 if len(v) else 0 for v in vals]   # convert to µ$
        bars = ax.bar(x + (idx - 1.5) * w, vals, w,
                      label=f"{prec.upper()} batch={bs}", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concurrencies])
    ax.set_xlabel("Concurrency (users)")
    ax.set_ylabel("Cost per image (µ$)")
    ax.set_title("Cloud Run Cost per Image by Configuration",
                 fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, os.path.join(output_dir, "cost_per_image.png"))


def plot_cost_vs_throughput(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: cost-per-image vs images-per-second (higher RPS × lower cost = better)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Cost-Efficiency: Cost per Image vs. Throughput",
                 fontweight="bold")

    for (prec, bs), grp in df.groupby(["precision", "batch_size"]):
        key   = f"{prec}_b{bs}"
        color = COLORS.get(key, "#888")
        label = f"{prec.upper()} batch={bs}"
        img_per_s = grp["rps"] * grp["batch_size"]
        cost_uc   = grp["cost_per_image"] * 1e6   # µ$
        ax.scatter(img_per_s, cost_uc, color=color, label=label, s=80, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(
                f"c={int(row['concurrency'])}",
                (row["rps"] * row["batch_size"], row["cost_per_image"] * 1e6),
                textcoords="offset points", xytext=(5, 3), fontsize=7,
            )

    ax.set_xlabel("Images / second (RPS × batch_size)")
    ax.set_ylabel("Cost per image (µ$)")
    ax.legend(fontsize=9)
    save(fig, os.path.join(output_dir, "cost_vs_throughput.png"))


def plot_fp32_vs_int8_cost(df: pd.DataFrame, output_dir: str) -> None:
    """Side-by-side bars: FP32 vs INT8 cost-per-image at batch=1."""
    sub = df[df["batch_size"] == 1].sort_values("concurrency")
    concurrencies = sorted(sub["concurrency"].unique())
    x = np.arange(len(concurrencies))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("FP32 vs INT8 Cost per Image (batch=1)", fontweight="bold")

    for i, (prec, color) in enumerate([("fp32", COLORS["fp32_b1"]),
                                        ("int8", COLORS["int8_b1"])]):
        vals = [
            sub[(sub["precision"] == prec) & (sub["concurrency"] == c)
                ]["cost_per_image"].values
            for c in concurrencies
        ]
        vals = [v[0] * 1e6 if len(v) else 0 for v in vals]
        ax.bar(x + (i - 0.5) * w, vals, w,
               label=prec.upper(), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concurrencies])
    ax.set_xlabel("Concurrency (users)")
    ax.set_ylabel("Cost per image (µ$)")
    ax.legend()

    # Annotate INT8/FP32 cost ratio above INT8 bars
    for i, c in enumerate(concurrencies):
        fp32_row = sub[(sub["precision"] == "fp32") & (sub["concurrency"] == c)]
        int8_row = sub[(sub["precision"] == "int8") & (sub["concurrency"] == c)]
        if not fp32_row.empty and not int8_row.empty:
            ratio = int8_row["cost_per_image"].values[0] / fp32_row["cost_per_image"].values[0]
            y_pos = int8_row["cost_per_image"].values[0] * 1e6
            ax.annotate(f"{ratio:.1f}×\ncostlier",
                        xy=(x[i] + 0.5 * w, y_pos),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color="darkred", fontweight="bold")

    save(fig, os.path.join(output_dir, "cost_fp32_vs_int8.png"))


# ── Console summary table ──────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    cols = ["label", "rps", "avg_ms", "failures",
            "cost_per_request", "cost_per_image", "cost_per_1k_images",
            "images_per_dollar", "hourly_cost_usd"]
    out = df[cols].copy()
    out["cost_per_request"]  = out["cost_per_request"].map("${:.6f}".format)
    out["cost_per_image"]    = out["cost_per_image"].map("${:.6f}".format)
    out["cost_per_1k_images"]= out["cost_per_1k_images"].map("${:.4f}".format)
    out["images_per_dollar"] = out["images_per_dollar"].map("{:,.0f}".format)
    out["hourly_cost_usd"]   = out["hourly_cost_usd"].map("${:.4f}".format)
    out["avg_ms"]            = out["avg_ms"].map("{:.0f}ms".format)
    out["rps"]               = out["rps"].map("{:.1f}".format)
    out["failures"]          = out["failures"].map("{:.0f}".format)
    print("\n" + out.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud Run cost analysis")
    parser.add_argument("--summary",    default="results/cloud_run_cpu/charts/summary.json")
    parser.add_argument("--output-dir", default="results/cloud_run_cpu/charts")
    args = parser.parse_args()

    print(f"Loading benchmark data from {args.summary}…")
    with open(args.summary) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    print(f"\nCloud Run pricing used:")
    print(f"  CPU:      ${CPU_PRICE_PER_VCPU_S}/vCPU-second  × {CPU_COUNT} vCPU")
    print(f"  Memory:   ${MEM_PRICE_PER_GIB_S}/GiB-second   × {MEMORY_GIB} GiB")
    print(f"  Requests: ${REQUEST_PRICE:.7f}/request")

    df = compute_costs(df)

    print("\n── Cost breakdown per configuration ──────────────────────────────────")
    print_summary(df)

    # Highlight best and worst configurations
    valid = df[df["failures"] == 0]
    if not valid.empty:
        cheapest = valid.loc[valid["cost_per_image"].idxmin()]
        priciest = valid.loc[valid["cost_per_image"].idxmax()]
        print(f"\n  Cheapest (no failures): {cheapest['label']}"
              f"  → ${cheapest['cost_per_image']:.6f}/image"
              f"  ({cheapest['images_per_dollar']:,.0f} images/$)")
        print(f"  Priciest (no failures): {priciest['label']}"
              f"  → ${priciest['cost_per_image']:.6f}/image"
              f"  ({priciest['images_per_dollar']:,.0f} images/$)")

    print("\nGenerating charts…")
    os.makedirs(args.output_dir, exist_ok=True)
    plot_cost_per_image(df, args.output_dir)
    plot_cost_vs_throughput(df, args.output_dir)
    plot_fp32_vs_int8_cost(df, args.output_dir)

    # Persist enriched data
    out_path = os.path.join(args.output_dir, "cost_analysis.json")
    df.to_json(out_path, orient="records", indent=2)
    print(f"  Saved → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
