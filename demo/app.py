"""
CIFAR-100 Inference Service – Live Demo
COMSE6998 · Applied Machine Learning in the Cloud

Usage:
    CIFAR100_API_URL=https://your-cloud-run-url streamlit run demo/app.py
"""
import os
import io
import base64
import time
import concurrent.futures

import numpy as np
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CIFAR-100 Inference Service",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "cloud_run_cpu")
CHARTS_DIR   = os.path.join(RESULTS_DIR, "charts")

API_URL_DEFAULT = "https://cifar100-service-417611172850.europe-west4.run.app"

# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def predict(api_url: str, b64: str, precision: str) -> dict:
    resp = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"image_b64": b64, "model_precision": precision, "return_profile": True},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def health(api_url: str) -> tuple[dict | None, str]:
    """Returns (response_dict, error_message). error_message is "" on success."""
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=15)
        if r.status_code == 200:
            return r.json(), ""
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.Timeout:
        return None, "Timed out after 15s — service may be cold-starting (minScale=0). Try again in 10s."
    except requests.exceptions.ConnectionError:
        return None, "Connection refused — check the URL."
    except Exception as exc:
        return None, str(exc)

def show_chart(path: str, caption: str = ""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Chart not found: {path}")

def confidence_bars(predictions: list, title: str, color: str) -> plt.Figure:
    labels = [p["class_name"].replace("_", " ") for p in predictions]
    values = [p["confidence"] * 100 for p in predictions]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(labels[::-1], values[::-1], color=color, alpha=0.85)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    for i, val in enumerate(values[::-1]):
        ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def latency_histogram(fp32_lats: list, int8_lats: list, n: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, lats, label, color in zip(
        axes,
        [fp32_lats, int8_lats],
        ["FP32", "INT8"],
        ["#4C72B0", "#DD8452"],
    ):
        if lats:
            ax.hist(lats, bins=min(20, len(lats)), color=color, alpha=0.85, edgecolor="white")
            ax.axvline(np.percentile(lats, 50), color="white", linestyle="--", linewidth=1.5, label=f"p50 {np.percentile(lats, 50):.0f}ms")
            ax.axvline(np.percentile(lats, 95), color="yellow", linestyle="--", linewidth=1.5, label=f"p95 {np.percentile(lats, 95):.0f}ms")
            ax.legend(fontsize=8)
        ax.set_title(f"{label} — {len(lats)} requests", fontweight="bold")
        ax.set_xlabel("Latency (ms)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Count")
    fig.suptitle(f"Latency Distribution — {n} concurrent requests", fontweight="bold")
    fig.tight_layout()
    return fig

def run_burst(api_url: str, b64: str, precision: str, n: int):
    """Fire n concurrent requests and return (latencies_ms, failures)."""
    def fire(_):
        t0 = time.perf_counter()
        try:
            predict(api_url, b64, precision)
            return (time.perf_counter() - t0) * 1000, True
        except Exception:
            return (time.perf_counter() - t0) * 1000, False

    t_wall = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(fire, range(n)))
    wall_ms = (time.perf_counter() - t_wall) * 1000

    lats = [r[0] for r in results if r[1]]
    return lats, wall_ms

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("CIFAR-100 Inference")
    st.caption("COMSE6998 · Applied ML in the Cloud  \nColumbia University")
    st.divider()

    api_url = st.text_input("API URL", value=API_URL_DEFAULT)

    if st.button("Check service health", use_container_width=True):
        with st.spinner("Checking…"):
            h, err = health(api_url)
        if h:
            st.success("Service is live")
            loaded = h.get("models_loaded", [])
            st.write(f"- FP32: {'✓' if 'fp32' in loaded else '✗'}")
            st.write(f"- INT8: {'✓' if 'int8' in loaded else '✗'}")
            st.write(f"- Device: `{h.get('device', 'unknown')}`")
        else:
            st.error(err)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("End-to-End ML Inference Service on GCP")
st.caption(
    "EfficientNet-B0 fine-tuned on CIFAR-100 · FastAPI · Cloud Run · "
    "FP32 vs INT8 Quantization · Locust Load Testing"
)

q1, q2, q3 = st.columns(3)
q1.info("**RQ1:** Does INT8 quantization improve latency without hurting accuracy?")
q2.info("**RQ2:** How does CPU vs GPU instance impact throughput?")
q3.info("**RQ3:** At what concurrency level does the service become a bottleneck?")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_live, tab_load, tab_quant, tab_gpu, tab_scale, tab_cost, tab_arch = st.tabs([
    "🚀  Live Demo",
    "🔥  Live Load Test",
    "⚡  RQ1 · FP32 vs INT8 (CPU)",
    "🖥️  RQ2 · GPU vs CPU",
    "📈  RQ3 · Scalability",
    "💰  Cost Analysis",
    "🏗️  Architecture",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Live Demo
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("Live Inference — FP32 vs INT8")
    st.caption("Upload any image. Both model variants are called simultaneously.")

    col_up, col_fp32, col_int8 = st.columns([1, 1.5, 1.5])

    with col_up:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)

    if uploaded and st.button("Run FP32 + INT8 →", type="primary", use_container_width=True):
        b64 = encode_image(img)
        with st.spinner("Sending to Cloud Run…"):
            try:
                fp32_res = predict(api_url, b64, "fp32")
                int8_res = predict(api_url, b64, "int8")

                fp32_ms = fp32_res["inference_ms"]
                int8_ms = int8_res["inference_ms"]
                speedup = fp32_ms / int8_ms if int8_ms > 0 else 1.0

                m1, m2, m3 = st.columns(3)
                m1.metric("FP32 latency", f"{fp32_ms:.1f} ms")
                m2.metric("INT8 latency", f"{int8_ms:.1f} ms")
                delta = f"{(speedup-1)*100:.0f}% faster" if speedup > 1 else f"{(1-speedup)*100:.0f}% slower"
                m3.metric("INT8 speedup", f"{speedup:.2f}×", delta=delta)

                if fp32_res.get("profile"):
                    p = fp32_res["profile"]
                    st.caption("FP32 breakdown")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Preprocess", f"{p['preprocess_ms']:.1f} ms")
                    c2.metric("Forward pass", f"{p['forward_ms']:.1f} ms")
                    c3.metric("Postprocess", f"{p['postprocess_ms']:.1f} ms")

                with col_fp32:
                    st.subheader("FP32 predictions")
                    fig = confidence_bars(fp32_res["predictions"][:5], "FP32 — Top 5", "#4C72B0")
                    st.pyplot(fig); plt.close(fig)

                with col_int8:
                    st.subheader("INT8 predictions")
                    fig = confidence_bars(int8_res["predictions"][:5], "INT8 — Top 5", "#DD8452")
                    st.pyplot(fig); plt.close(fig)

                top_fp32 = fp32_res["predictions"][0]["class_name"].replace("_", " ").title()
                top_int8 = int8_res["predictions"][0]["class_name"].replace("_", " ").title()
                if top_fp32 == top_int8:
                    st.success(f"Both models agree → **{top_fp32}**")
                else:
                    st.warning(f"Models disagree — FP32: **{top_fp32}** · INT8: **{top_int8}**")

            except Exception as exc:
                st.error(f"Request failed: {exc}")
    elif not uploaded:
        st.info("Upload an image to run a live prediction.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Live Load Test
# ══════════════════════════════════════════════════════════════════════════════
with tab_load:
    st.header("Live Load Test")
    st.caption(
        "Fire a burst of concurrent requests to the live endpoint. "
        "Run at c=5 → c=25 → c=50 to watch latency grow — this is exactly what the Locust benchmark measured."
    )

    st.info(
        "**Why is latency higher here than in the Live Demo tab?**  \n"
        "The server runs 2 Uvicorn workers (synchronous handlers). "
        "When you fire N concurrent requests, only 2 run in parallel — the rest queue. "
        "At c=10: ~5 requests queue behind 2 workers × ~50ms each = expected p95 of 250–500ms. "
        "This queuing effect IS the scalability bottleneck the project measures. "
        "**Run the warm-up first** to avoid cold-start penalty on the first burst."
    )

    col_cfg, col_img = st.columns([2, 1])

    with col_cfg:
        lt_users = st.select_slider(
            "Concurrent requests",
            options=[5, 10, 25, 50, 100],
            value=10,
        )
        lt_precision = st.radio("Model precision", ["fp32", "int8"], horizontal=True, key="lt_prec")
        st.caption(
            f"Will fire **{lt_users}** simultaneous POST /predict requests "
            f"using the **{lt_precision.upper()}** model."
        )

    with col_img:
        lt_uploaded = st.file_uploader("Test image", type=["jpg", "jpeg", "png"], key="lt_img")
        if lt_uploaded:
            st.image(Image.open(lt_uploaded), use_container_width=True)

    if lt_uploaded and st.button("🔥 Warm up service first (send 1 request)", use_container_width=True):
        with st.spinner("Warming up…"):
            try:
                b64_wu = encode_image(Image.open(lt_uploaded))
                r = predict(api_url, b64_wu, "fp32")
                st.success(f"Service warm — FP32 latency: {r['inference_ms']:.1f} ms. Ready for load test.")
            except Exception as exc:
                st.error(f"Warm-up failed: {exc}")

    compare_mode = st.checkbox("Compare FP32 vs INT8 side by side", value=True)

    if lt_uploaded:
        btn_label = (
            f"Fire {lt_users} concurrent requests (FP32 + INT8)"
            if compare_mode
            else f"Fire {lt_users} concurrent {lt_precision.upper()} requests"
        )
        if st.button(btn_label, type="primary", use_container_width=True):
            b64_lt = encode_image(Image.open(lt_uploaded))

            def fmt_ms(ms: float) -> str:
                return f"{ms/1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"

            if compare_mode:
                with st.spinner(f"Running {lt_users} concurrent FP32 requests…"):
                    fp32_lats, fp32_wall = run_burst(api_url, b64_lt, "fp32", lt_users)
                with st.spinner(f"Running {lt_users} concurrent INT8 requests…"):
                    int8_lats, int8_wall = run_burst(api_url, b64_lt, "int8", lt_users)

                st.subheader("Results")

                if fp32_lats:
                    fa = np.array(fp32_lats)
                    st.markdown("**FP32**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("p50", fmt_ms(np.percentile(fa, 50)))
                    c2.metric("p95", fmt_ms(np.percentile(fa, 95)))
                    c3.metric("p99", fmt_ms(np.percentile(fa, 99)))
                    c4.metric("RPS", f"{len(fp32_lats)/(fp32_wall/1000):.1f}")

                if int8_lats:
                    ia = np.array(int8_lats)
                    st.markdown("**INT8**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("p50", fmt_ms(np.percentile(ia, 50)))
                    c2.metric("p95", fmt_ms(np.percentile(ia, 95)))
                    c3.metric("p99", fmt_ms(np.percentile(ia, 99)))
                    c4.metric("RPS", f"{len(int8_lats)/(int8_wall/1000):.1f}")

                # Bar chart + histograms
                if fp32_lats and int8_lats:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    x, w = np.arange(2), 0.3
                    ax.bar(x - w/2, [np.percentile(fa, 50), np.percentile(fa, 95)],
                           w, label="FP32", color="#4C72B0", alpha=0.85)
                    ax.bar(x + w/2, [np.percentile(ia, 50), np.percentile(ia, 95)],
                           w, label="INT8", color="#DD8452", alpha=0.85)
                    ax.set_xticks(x); ax.set_xticklabels(["p50", "p95"])
                    ax.set_ylabel("Latency (ms)")
                    ax.set_title(f"FP32 vs INT8 — {lt_users} concurrent requests", fontweight="bold")
                    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                    fig = latency_histogram(fp32_lats, int8_lats, lt_users)
                    st.pyplot(fig); plt.close(fig)

            else:
                with st.spinner(f"Firing {lt_users} concurrent {lt_precision.upper()} requests…"):
                    lats, wall_ms = run_burst(api_url, b64_lt, lt_precision, lt_users)

                st.subheader("Results")
                if lats:
                    arr = np.array(lats)
                    rps = len(lats) / (wall_ms / 1000)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("p50", fmt_ms(np.percentile(arr, 50)))
                    m2.metric("p95", fmt_ms(np.percentile(arr, 95)))
                    m3.metric("p99", fmt_ms(np.percentile(arr, 99)))
                    m4.metric("RPS", f"{rps:.1f}")

                    color = "#4C72B0" if lt_precision == "fp32" else "#DD8452"
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.hist(arr, bins=min(20, len(arr)), color=color, alpha=0.85, edgecolor="white")
                    ax.axvline(np.percentile(arr, 50), color="white",  linestyle="--", lw=1.5,
                               label=f"p50 {fmt_ms(np.percentile(arr, 50))}")
                    ax.axvline(np.percentile(arr, 95), color="yellow", linestyle="--", lw=1.5,
                               label=f"p95 {fmt_ms(np.percentile(arr, 95))}")
                    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
                    ax.set_title(f"{lt_precision.upper()} — {lt_users} concurrent requests", fontweight="bold")
                    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                else:
                    st.warning("No responses received — check the service URL.")

    else:
        st.info("Upload a test image above to begin.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – RQ1 · FP32 vs INT8 CPU
# ══════════════════════════════════════════════════════════════════════════════
with tab_quant:
    st.header("RQ1 · FP32 CPU vs INT8 CPU — Does quantization improve latency?")

    st.subheader("Latency Comparison — CPU-only (batch=1, concurrency=10)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("FP32 p50", "430 ms")
    a2.metric("INT8 p50", "250 ms", delta="-180 ms faster")
    a3.metric("Speedup", "1.72×", delta="INT8 faster on CPU")
    a4.metric("Accuracy preserved", "86.4% → 86.5%", delta="+0.1pp (INT8 ≥ FP32)")
    st.caption(
        "Cloud Run CPU (2 vCPU, 2 GiB) · batch=1 · concurrency=10. "
        "INT8 post-training dynamic quantization (Linear + Conv2d) via `torch.quantization.quantize_dynamic`. "
        "Speedup holds across all concurrency levels (1.72× at c=10, c=50, c=200)."
    )

    st.divider()
    st.subheader("Latency & Throughput — CPU-only Benchmark")
    st.caption("Cloud Run CPU (2 vCPU, 2 GiB) · batch=1 · concurrency = 10, 50, 200")

    show_chart(os.path.join(CHARTS_DIR, "fp32_cpu_vs_int8_cpu.png"))

    col_a, col_b = st.columns(2)
    with col_a:
        show_chart(os.path.join(CHARTS_DIR, "speedup_quantization.png"),
                   "p50/p95 latency vs concurrency (all configs)")
    with col_b:
        show_chart(os.path.join(CHARTS_DIR, "concurrency_heatmap.png"),
                   "p95 latency heatmap across precision × batch × concurrency")

    st.success(
        "**Finding — RQ1:** On CPU, INT8 is consistently **~1.7× faster** than FP32 at batch=1 "
        "(250ms vs 430ms p50 at c=10). "
        "INT8 throughput reaches 42 RPS vs FP32's 25 RPS at c=50.  \n"
        "Accuracy is perfectly preserved (86.4% → 86.5%).  \n"
        "**Conclusion:** INT8 dynamic quantization (Linear + Conv2d) delivers meaningful latency "
        "and throughput gains on CPU Cloud Run with no accuracy cost."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – RQ2 · GPU vs CPU
# ══════════════════════════════════════════════════════════════════════════════
with tab_gpu:
    st.header("RQ2 · FP32 GPU vs FP32 CPU — How much does the hardware matter?")
    st.caption(
        "GPU deployment: Cloud Run with NVIDIA T4 (device=cuda) · "
        "CPU deployment: Cloud Run 2 vCPU 2 GiB · both running FP32 model"
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("GPU FP32 p50 @ c=10", "160 ms")
    g2.metric("CPU FP32 p50 @ c=10", "430 ms")
    g3.metric("GPU speedup", "2.7×", delta="faster at low concurrency")

    show_chart(os.path.join(CHARTS_DIR, "fp32_gpu_vs_fp32_cpu.png"))

    st.info(
        "**Finding — RQ2:** GPU (T4) is **2.7× faster** than CPU at low concurrency (c=10) "
        "and **2.2× faster** at higher concurrency (c=50, c=200) for FP32 batch=1.  \n"
        "GPU throughput saturates at **~50 RPS** (memory-bandwidth limited); "
        "CPU saturates at **~23 RPS** (compute limited).  \n"
        "**Conclusion:** GPU eliminates the per-request compute bottleneck but both "
        "deployments hit a throughput ceiling — scale horizontally via Cloud Run replicas."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – RQ3 · Scalability
# ══════════════════════════════════════════════════════════════════════════════
with tab_scale:
    st.header("RQ3 · At what concurrency level does the service become a bottleneck?")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("FP32 CPU b=1 peak RPS", "~25", help="Saturates at c=10; CPU compute bound")
    s2.metric("INT8 CPU b=1 peak RPS", "~42", help="INT8 is 1.7× faster, more throughput per instance")
    s3.metric("INT8 b=8 p50 @ c=200", "28 s", help="Batch + high concurrency = queue collapse on CPU")
    s4.metric("Failures across all cells", "0", help="Clean run — no timeouts with 300s limit")

    col_c, col_d = st.columns(2)
    with col_c:
        show_chart(os.path.join(CHARTS_DIR, "latency_throughput.png"),
                   "Latency vs Throughput — all 12 configurations")
    with col_d:
        show_chart(os.path.join(CHARTS_DIR, "batch_comparison.png"),
                   "Batch=1 vs Batch=8 throughput (RPS)")

    st.info(
        "**Finding — RQ3:** FP32 CPU batch=1 throughput **saturates at ~23 RPS**; "
        "INT8 CPU batch=1 reaches **42 RPS** — 1.8× more throughput from quantization alone. "
        "Both plateau because a single Cloud Run instance is CPU-compute limited. "
        "Batch=8 at c=200 collapses for both precisions (28–31s p50) — "
        "the request queue grows faster than the 2 Uvicorn workers can drain it. "
        "Auto-scaling adds replicas but each remains CPU-limited per request."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Cost Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_cost:
    st.header("Cloud Run Cost Analysis")
    st.caption(
        "Always-allocated pricing · 2 vCPU · 2 GiB  \n"
        "CPU $0.000024/vCPU-s · Memory $0.0000025/GiB-s · Requests $0.40/million"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Cheapest config", "int8_b8_c10")
    c2.metric("Efficiency", "94,854 images/$", help="INT8 batch=8 at low concurrency — best cost-efficiency")
    c3.metric("INT8 vs FP32 cost", "~10% cheaper", help="INT8 is faster on CPU → shorter CPU hold → lower cost")

    col_e, col_f = st.columns(2)
    with col_e:
        show_chart(os.path.join(CHARTS_DIR, "cost_per_image.png"))
    with col_f:
        show_chart(os.path.join(CHARTS_DIR, "cost_fp32_vs_int8.png"))

    show_chart(os.path.join(CHARTS_DIR, "cost_vs_throughput.png"))

    st.success(
        "**Finding:** On CPU, INT8 is **cheaper** than FP32 per image — "
        "it is faster, so it holds the CPU for less time per request. "
        "Best config: **INT8 batch=8 at c=10** (94,854 images/$) where "
        "batch amortisation and INT8 speed combine without queue buildup."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.header("System Architecture")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Model")
        st.markdown("""
| | |
|---|---|
| Architecture | EfficientNet-B0 |
| Dataset | CIFAR-100 (100 classes) |
| Training | 2-phase fine-tuning, 25 epochs |
| Phase 1 (1–5) | Head only, backbone frozen |
| Phase 2 (6–25) | Full fine-tune, cosine LR annealing |
| Augmentation | RandomResizedCrop (scale 0.5–1.0) |
| FP32 Top-1 | **86.4%** |
| INT8 (dynamic, Linear + Conv2d) | **86.5%** |
""")

        st.subheader("Inference API")
        st.markdown("""
| | |
|---|---|
| Framework | FastAPI + Uvicorn |
| Endpoints | `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /metrics` |
| Max batch | 8 images |
| Profiling | Per-request preprocess / forward / postprocess breakdown |
| Monitoring | GCP Cloud Monitoring — 4 custom metrics, 60s flush |
""")

    with col_right:
        st.subheader("GCP Infrastructure")
        st.markdown("""
| | |
|---|---|
| Model storage | GCS bucket |
| Serving | Cloud Run (serverless) |
| Auto-scaling | 0 → 10 replicas (target concurrency = 5) |
| Resources | 2 vCPU · 2 GiB RAM per instance |
| Timeout | 300 s |
| CI/CD | Cloud Build |
""")

        st.subheader("Benchmarking")
        st.markdown("""
| | |
|---|---|
| Tool | Locust |
| Matrix | precision × batch × concurrency = **12 cells** |
| Precision | fp32, int8 |
| Batch sizes | 1, 8 |
| Concurrency | 10, 50, 200 users |
| Metrics | p50/p95/p99 latency · RPS · failure rate · cost/image |
""")

    st.divider()
    st.subheader("Key Findings")
    f1, f2, f3 = st.columns(3)
    f1.success(
        "**Quantization (RQ1):** INT8 is **1.7× faster** than FP32 on CPU "
        "(250ms vs 430ms p50 at c=10) with accuracy perfectly preserved "
        "(86.4% → 86.5%)."
    )
    f2.info(
        "**GPU vs CPU (RQ2):** GPU (T4) is **2.7× faster** than CPU for FP32 "
        "(160ms vs 430ms). GPU: ~50 RPS. CPU: ~23 RPS."
    )
    f3.warning(
        "**Bottleneck (RQ3):** CPU saturates at ~23 RPS (FP32) / ~42 RPS (INT8). "
        "Batch=8 at c=200 collapses (28–31s p50). "
        "Best config: INT8 batch=8 c=10 — 94,854 images/$."
    )
