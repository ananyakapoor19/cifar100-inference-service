# Cost Analysis Report — Cloud Run CPU vs GCE GPU (T4)

**Model:** EfficientNet-B0 fine-tuned on CIFAR-100  
**Region:** us-central1  
**Date:** 2026-05-05

---

## 1. Pricing Models

### Cloud Run CPU (always-allocated)

| Resource | Rate | Allocation |
|---|---|---|
| CPU | $0.00002400 / vCPU-second | 2 vCPU |
| Memory | $0.00000250 / GiB-second | 2 GiB |
| Requests | $0.0000004 / request | — |

Cost model: **pay-per-millisecond of request duration**. Idle time between requests is not billed.

### GCE GPU — n1-standard-4 + NVIDIA T4 (on-demand)

| Resource | Rate |
|---|---|
| n1-standard-4 (4 vCPU, 15 GB) | $0.190004 / hour |
| NVIDIA T4 GPU | $0.350000 / hour |
| **Total** | **$0.540004 / hour ($0.00015001 / second)** |

Cost model: **flat hourly rate**. VM runs continuously regardless of traffic; effective per-request cost scales with latency.

---

## 2. Full Cost Breakdown

### Cloud Run CPU — All Configurations

| Config | RPS | Avg Latency | p50 | p95 | Failures | $/request | $/image | $/1k images | Images/$ | Hourly $ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 · b=1 · c=10 | 22.6 | 411ms | 430ms | 470ms | 0 | $0.000022 | $0.000022 | $0.0222 | 45,042 | $1.81 |
| FP32 · b=1 · c=50 | 24.7 | 1,894ms | 1,900ms | 2,200ms | 0 | $0.000101 | $0.000101 | $0.1008 | 9,920 | $8.96 |
| FP32 · b=1 · c=200 | 23.2 | 6,623ms | 7,800ms | 8,200ms | 0 | $0.000351 | $0.000351 | $0.3514 | 2,846 | $29.33 |
| FP32 · b=8 · c=10 | 5.4 | 1,758ms | 1,700ms | 1,900ms | 0 | $0.000094 | $0.000012 | $0.0117 | 85,474 | $1.82 |
| FP32 · b=8 · c=50 | 5.1 | 8,812ms | 8,800ms | 11,000ms | 0 | $0.000467 | $0.000058 | $0.0584 | 17,114 | $8.60 |
| FP32 · b=8 · c=200 | 3.5 | 31,518ms | 33,000ms | 38,000ms | 0 | $0.001671 | $0.000209 | $0.2089 | 4,788 | $21.31 |
| INT8 · b=1 · c=10 | 24.5 | 376ms | 250ms | 310ms | 0 | $0.000020 | $0.000020 | $0.0203 | 49,154 | $1.79 |
| INT8 · b=1 · c=50 | 42.1 | 1,111ms | 1,100ms | 1,400ms | 0 | $0.000059 | $0.000059 | $0.0593 | 16,873 | $8.98 |
| INT8 · b=1 · c=200 | 40.4 | 3,935ms | 4,500ms | 5,200ms | 0 | $0.000209 | $0.000209 | $0.2089 | 4,786 | $30.43 |
| INT8 · b=8 · c=10 | 6.0 | 1,584ms | 1,500ms | 1,800ms | 0 | $0.000084 | $0.000011 | $0.0105 | 94,854 | $1.81 |
| INT8 · b=8 · c=50 | 5.7 | 7,684ms | 7,700ms | 10,000ms | 0 | $0.000408 | $0.000051 | $0.0510 | 19,624 | $8.43 |
| INT8 · b=8 · c=200 | 4.1 | 26,645ms | 28,000ms | 36,000ms | 0 | $0.001413 | $0.000177 | $0.1766 | 5,663 | $20.83 |

### GCE GPU (T4) — All Configurations

> Note: fp32_b8_c10 and fp32_b8_c200 had failures (9 and 133 respectively) and are excluded from best-config recommendations.

| Config | RPS | Avg Latency | p50 | p95 | Failures | $/request | $/image | $/1k images | Images/$ | Hourly $ |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 · b=1 · c=10 | 46.9 | 181ms | 160ms | 260ms | 0 | $0.000027 | $0.000027 | $0.0272 | 36,733 | $0.54 |
| FP32 · b=1 · c=50 | 50.2 | 923ms | 860ms | 1,400ms | 0 | $0.000138 | $0.000138 | $0.1385 | 7,222 | $0.54 |
| FP32 · b=1 · c=200 | 51.6 | 3,094ms | 3,500ms | 4,100ms | 0 | $0.000464 | $0.000464 | $0.4641 | 2,155 | $0.54 |
| FP32 · b=8 · c=10 | 12.2 | 740ms | 430ms | 2,000ms | **9** | $0.000111 | $0.000014 | $0.0139 | 72,065 | $0.54 |
| FP32 · b=8 · c=50 | 28.1 | 1,625ms | 1,600ms | 2,500ms | 0 | $0.000244 | $0.000030 | $0.0305 | 32,810 | $0.54 |
| FP32 · b=8 · c=200 | 28.1 | 5,396ms | 5,900ms | 7,200ms | **133** | $0.000809 | $0.000101 | $0.1012 | 9,884 | $0.54 |

---

## 3. GPU vs CPU Head-to-Head (FP32 · batch=1)

| Concurrency | CPU p50 | GPU p50 | Latency Speedup | CPU RPS | GPU RPS | Throughput Gain | CPU $/img | GPU $/img | GPU Cost Premium |
|---|---|---|---|---|---|---|---|---|---|
| c=10 | 430ms | 160ms | **2.7×** | 22.6 | 46.9 | **2.1×** | $0.000022 | $0.000027 | +20% |
| c=50 | 1,900ms | 860ms | **2.2×** | 24.7 | 50.2 | **2.0×** | $0.000101 | $0.000138 | +37% |
| c=200 | 7,800ms | 3,500ms | **2.2×** | 23.2 | 51.6 | **2.2×** | $0.000351 | $0.000464 | +32% |

**Verdict:** The GPU is consistently 2–2.7× faster in latency and 2× higher in throughput, but costs 1.2–1.4× more per image. The gap widens at higher concurrency because Cloud Run's per-request billing keeps CPU cost proportional to latency while the GPU's flat $0.54/hr is spread over more requests only when throughput is high enough.

---

## 4. INT8 CPU vs GPU FP32 (batch=1 · c=10 — the practical sweet spot)

| Metric | INT8 CPU | GPU FP32 |
|---|---|---|
| p50 latency | 250ms | 160ms |
| p95 latency | 310ms | 260ms |
| RPS | 24.5 | 46.9 |
| $/image | $0.000020 | $0.000027 |
| Images/$ | 49,154 | 36,733 |
| Hourly cost | $1.79 | $0.54 (flat) |

INT8 quantization on CPU partially closes the latency gap with GPU (250ms vs 160ms, only 1.6× difference) while remaining **1.3× cheaper per image**. For cost-sensitive workloads that can tolerate ~250ms, INT8 CPU is the better choice.

---

## 5. Effect of Batching on CPU

Batching amortises the fixed per-request overhead across multiple images, dramatically improving cost efficiency:

| Config | $/image | Images/$ | vs FP32·b=1·c=10 |
|---|---|---|---|
| FP32 · b=1 · c=10 | $0.000022 | 45,042 | baseline |
| FP32 · b=8 · c=10 | $0.000012 | 85,474 | **1.9× cheaper** |
| INT8 · b=1 · c=10 | $0.000020 | 49,154 | 1.1× cheaper |
| INT8 · b=8 · c=10 | $0.000011 | 94,854 | **2.1× cheaper** |

The GPU batch=8 results are less conclusive due to failures at c=10 and c=200; only c=50 ran cleanly (32,810 images/$).

---

## 6. Hourly Cost at Scale

Cloud Run CPU hourly cost varies with observed RPS (pay-per-request), while GCE GPU is always $0.54/hr:

| Scenario | Platform | Config | Hourly Cost |
|---|---|---|---|
| Low traffic · low latency | Cloud Run CPU | INT8 b=1 c=10 | $1.79 |
| Low traffic · low latency | GCE GPU | FP32 b=1 c=10 | **$0.54 (flat)** |
| High traffic burst | Cloud Run CPU | INT8 b=1 c=200 | $30.43 |
| High traffic burst | GCE GPU | FP32 b=1 c=200 | **$0.54 (flat)** |

At high concurrency, the GPU's flat rate becomes dramatically cheaper in hourly terms — but only if the VM is kept busy. At idle or low traffic, Cloud Run's serverless model (scales to 0) is the better choice since you pay nothing when no requests arrive.

---

## 7. Best Configuration by Use Case

| Use Case | Recommended Config | $/image | Notes |
|---|---|---|---|
| **Maximum cost efficiency** | CPU · INT8 · b=8 · c=10 | $0.000011 | 94,854 images/$ — best overall |
| **Lowest latency** | GPU · FP32 · b=1 · c=10 | $0.000027 | 160ms p50, zero failures |
| **Highest throughput** | GPU · FP32 · b=8 · c=50 | $0.000030 | 225 images/s, zero failures |
| **Latency + cost balance** | CPU · INT8 · b=1 · c=10 | $0.000020 | 250ms p50, 49,154 images/$ |
| **Sustained high traffic** | GPU · FP32 · b=1 · any | $0.54/hr flat | Flat rate pays off above ~35 req/s continuously |

---

## 8. Charts

| Chart | Description |
|---|---|
| `cost_cpu_vs_gpu.png` | Cost per image (µ$) grouped by concurrency across all configs |
| `images_per_dollar_comparison.png` | Cost efficiency (images/$) — higher is better |
| `latency_vs_cost.png` | Latency vs cost scatter — ideal configs are bottom-left |
| `throughput_latency_comparison.png` | RPS and p50 latency side-by-side, CPU vs GPU |

---

## 9. Consistency with Prior Analysis

The CPU cost numbers in this report were independently recomputed and match the existing `results/cloud_run_cpu/charts/cost_analysis.json` exactly across all 12 configurations. The GPU cost analysis (GCE n1-standard-4 + T4) is new — no prior GPU cost data existed in the repository.
