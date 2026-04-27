"""
plot_results.py – Parse Locust CSV outputs and produce publication-quality charts.

Charts generated:
  1. latency_throughput.png  – p50/p95/p99 latency vs. RPS per configuration
  2. concurrency_heatmap.png – p95 latency heatmap across precision × concurrency
  3. batch_comparison.png    – batch=1 vs batch=8 throughput bars
  4. speedup_quantization.png – FP32 vs INT8 latency across concurrency levels

Usage:
    python benchmarks/analysis/plot_results.py \
        --results-dir ./results \
        --output-dir  ./results/charts
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":     150,
    "font.family":    "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
})

COLORS = {
    "fp32_b1":  "#2196F3",
    "fp32_b8":  "#0D47A1",
    "int8_b1":  "#FF5722",
    "int8_b8":  "#BF360C",
}

CONCURRENCIES = [10, 50, 200]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_stats(results_dir: str) -> pd.DataFrame:
    """Load all Locust stats_stats.csv files into a single DataFrame."""
    rows = []
    pattern = os.path.join(results_dir, "*", "stats_stats.csv")
    for csv_path in sorted(glob.glob(pattern)):
        label = Path(csv_path).parent.name  # e.g. fp32_b1_c50
        parts = label.split("_")
        if len(parts) < 3:
            continue
        precision  = parts[0]
        batch_size = int(parts[1].lstrip("b"))
        concurrency= int(parts[2].lstrip("c"))

        try:
            df = pd.read_csv(csv_path)
            # Locust stats CSV has "Aggregated" row
            agg = df[df["Name"] == "Aggregated"]
            if agg.empty:
                agg = df.iloc[-1:]  # fallback: last row
            row = agg.iloc[0]
            rows.append({
                "precision":   precision,
                "batch_size":  batch_size,
                "concurrency": concurrency,
                "label":       label,
                "rps":         float(row.get("Requests/s", 0)),
                "failures":    float(row.get("Failure Count", 0)),
                "p50_ms":      float(row.get("50%", 0)),
                "p95_ms":      float(row.get("95%", 0)),
                "p99_ms":      float(row.get("99%", 0)),
                "avg_ms":      float(row.get("Average Response Time", 0)),
            })
        except Exception as e:
            print(f"  WARNING: could not parse {csv_path}: {e}")

    if not rows:
        print("No result CSV files found. Have you run run_experiments.sh yet?")
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Chart helpers ──────────────────────────────────────────────────────────────

def save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_latency_throughput(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Latency vs. Throughput by Configuration", fontsize=14, fontweight="bold")

    for ax, pct_col, pct_label in zip(axes,
                                       ["p50_ms", "p95_ms", "p99_ms"],
                                       ["p50 latency (ms)", "p95 latency (ms)", "p99 latency (ms)"]):
        for (prec, bs), grp in df.sort_values("concurrency").groupby(["precision", "batch_size"]):
            key   = f"{prec}_b{bs}"
            color = COLORS.get(key, "#888888")
            label = f"{prec.upper()} batch={bs}"
            ax.plot(grp["rps"], grp[pct_col],
                    marker="o", linewidth=2, color=color, label=label)
        ax.set_xlabel("Requests / s")
        ax.set_ylabel(pct_label)
        ax.set_title(pct_label)
        ax.legend(fontsize=8)

    save(fig, os.path.join(output_dir, "latency_throughput.png"))


def plot_concurrency_heatmap(df: pd.DataFrame, output_dir: str):
    """p95 latency heatmap: rows=precision×batch, cols=concurrency."""
    pivot = df.pivot_table(
        values="p95_ms",
        index=["precision", "batch_size"],
        columns="concurrency",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="p95 latency (ms)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"c={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{p.upper()} b={b}" for p, b in pivot.index])
    ax.set_title("p95 Latency Heatmap (ms) – precision × batch × concurrency",
                 fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color="black" if val < pivot.values.max() * 0.6 else "white",
                    fontsize=9, fontweight="bold")
    save(fig, os.path.join(output_dir, "concurrency_heatmap.png"))


def plot_batch_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart: RPS for batch=1 vs batch=8 at each concurrency."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Throughput: Batch=1 vs Batch=8", fontsize=13, fontweight="bold")

    for ax, prec in zip(axes, ["fp32", "int8"]):
        sub = df[df["precision"] == prec].sort_values(["concurrency", "batch_size"])
        x  = np.arange(len(CONCURRENCIES))
        w  = 0.35

        b1 = [sub[(sub["concurrency"] == c) & (sub["batch_size"] == 1)]["rps"].mean()
              for c in CONCURRENCIES]
        b8 = [sub[(sub["concurrency"] == c) & (sub["batch_size"] == 8)]["rps"].mean()
              for c in CONCURRENCIES]

        ax.bar(x - w/2, b1, w, label="batch=1",  color=COLORS.get(f"{prec}_b1", "#888"))
        ax.bar(x + w/2, b8, w, label="batch=8",  color=COLORS.get(f"{prec}_b8", "#555"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"c={c}" for c in CONCURRENCIES])
        ax.set_title(f"{prec.upper()}")
        ax.set_ylabel("Requests / s")
        ax.legend()

    save(fig, os.path.join(output_dir, "batch_comparison.png"))


def plot_quantization_speedup(df: pd.DataFrame, output_dir: str):
    """Line chart: FP32 vs INT8 p50 latency at batch=1 across concurrency."""
    sub = df[df["batch_size"] == 1].sort_values("concurrency")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Quantization Speedup: FP32 vs INT8 (batch=1)", fontweight="bold")

    for prec, color in [("fp32", COLORS["fp32_b1"]), ("int8", COLORS["int8_b1"])]:
        grp = sub[sub["precision"] == prec]
        ax.plot(grp["concurrency"], grp["p50_ms"],
                marker="o", linewidth=2.5, color=color, label=f"{prec.upper()} p50")
        ax.plot(grp["concurrency"], grp["p95_ms"],
                marker="s", linewidth=1.5, linestyle="--", color=color,
                alpha=0.7, label=f"{prec.upper()} p95")

    ax.set_xlabel("Concurrency (users)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(CONCURRENCIES)
    ax.legend()

    # Annotate speedup at each concurrency
    for c in CONCURRENCIES:
        fp32_row = sub[(sub["precision"] == "fp32") & (sub["concurrency"] == c)]
        int8_row = sub[(sub["precision"] == "int8") & (sub["concurrency"] == c)]
        if not fp32_row.empty and not int8_row.empty:
            speedup = fp32_row["p50_ms"].values[0] / max(int8_row["p50_ms"].values[0], 1e-6)
            y_pos   = int8_row["p50_ms"].values[0]
            ax.annotate(f"{speedup:.1f}×", xy=(c, y_pos),
                        xytext=(0, -18), textcoords="offset points",
                        ha="center", fontsize=9, color="darkgreen", fontweight="bold")

    save(fig, os.path.join(output_dir, "speedup_quantization.png"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir",  default="./results/charts")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}…")
    df = load_stats(args.results_dir)

    if df.empty:
        print("No data found. Exiting.")
        return

    print(f"Loaded {len(df)} experiment rows.")
    print(df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating charts…")
    plot_latency_throughput(df, args.output_dir)
    plot_concurrency_heatmap(df, args.output_dir)
    plot_batch_comparison(df, args.output_dir)
    plot_quantization_speedup(df, args.output_dir)

    # Also dump a summary JSON
    summary_path = os.path.join(args.output_dir, "summary.json")
    df.to_json(summary_path, orient="records", indent=2)
    print(f"  Saved → {summary_path}")
    print("\nAll charts generated successfully.")


if __name__ == "__main__":
    main()
