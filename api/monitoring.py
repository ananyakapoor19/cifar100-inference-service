"""
monitoring.py – GCP Cloud Monitoring integration for the inference service.

Buffers per-request metrics and flushes aggregated time-series data to
Cloud Monitoring every CM_FLUSH_INTERVAL seconds on a background thread.

Metrics pushed (custom.googleapis.com/cifar100_inference/...):
    request_count        – requests received per flush window       (INT64, GAUGE)
    error_count          – failed requests per flush window         (INT64, GAUGE)
    request_latency_ms   – mean end-to-end request latency          (DOUBLE, GAUGE)
    inference_latency_ms – mean model forward-pass latency          (DOUBLE, GAUGE)

Labels on every metric:
    model_precision  – fp32 | int8
    endpoint         – predict | predict_batch

Environment variables:
    GOOGLE_CLOUD_PROJECT   – GCP project ID; auto-fetched from the GCE/Cloud Run
                             metadata server if not set explicitly.
    CM_FLUSH_INTERVAL      – seconds between metric flushes (default: 60)

Graceful degradation:
    If GOOGLE_CLOUD_PROJECT cannot be resolved, or google-cloud-monitoring is
    not installed, or any write fails, monitoring silently no-ops so the
    inference server always stays healthy.
"""

import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "custom.googleapis.com/cifar100_inference"


# ── Per-window accumulator ─────────────────────────────────────────────────────

@dataclass
class _Bucket:
    count: int = 0
    errors: int = 0
    latency_sum: float = 0.0
    inference_sum: float = 0.0


class _MetricsBuffer:
    """Thread-safe accumulator for one flush window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, str], _Bucket] = defaultdict(_Bucket)

    def record(self, precision: str, endpoint: str,
               latency_ms: float, inference_ms: float, success: bool) -> None:
        key = (precision, endpoint)
        with self._lock:
            b = self._buckets[key]
            b.count += 1
            b.latency_sum += latency_ms
            b.inference_sum += inference_ms
            if not success:
                b.errors += 1

    def drain(self) -> Dict[Tuple[str, str], _Bucket]:
        """Atomically return and reset the current buffer."""
        with self._lock:
            snapshot = dict(self._buckets)
            self._buckets = defaultdict(_Bucket)
            return snapshot


# ── Cloud Monitoring reporter ──────────────────────────────────────────────────

class CloudMonitoringReporter:
    """
    Accumulates per-request metrics in memory and flushes aggregates to
    GCP Cloud Monitoring on a daemon background thread.

    Usage in FastAPI lifespan:
        reporter.start()   # begins background flush thread
        ...
        reporter.stop()    # final flush + thread join
    """

    def __init__(self, project_id: Optional[str] = None,
                 flush_interval: int = 60) -> None:
        self._flush_interval = flush_interval
        self._buffer = _MetricsBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = False

        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            project_id = self._fetch_project_from_metadata()

        if not project_id:
            logger.info(
                "Cloud Monitoring disabled: GOOGLE_CLOUD_PROJECT not set "
                "and metadata server unavailable (expected in local dev)."
            )
            return

        try:
            import google.cloud.monitoring_v3  # noqa: F401 – verify install
            self._project_id = project_id
            self._enabled = True
            logger.info(
                "Cloud Monitoring enabled (project=%s, flush_interval=%ds).",
                project_id, flush_interval,
            )
        except ImportError:
            logger.warning(
                "Cloud Monitoring disabled: google-cloud-monitoring not installed."
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="cm-flusher"
        )
        self._thread.start()
        logger.info("Cloud Monitoring flush thread started.")

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._do_flush()   # drain remaining buffer on graceful shutdown
        logger.info("Cloud Monitoring flush thread stopped.")

    def record(self, precision: str, endpoint: str,
               latency_ms: float, inference_ms: float, success: bool) -> None:
        """Record one request. Called from endpoint handlers; must not block."""
        if self._enabled:
            self._buffer.record(precision, endpoint, latency_ms, inference_ms, success)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        while not self._stop.wait(self._flush_interval):
            self._do_flush()

    def _do_flush(self) -> None:
        data = self._buffer.drain()
        if not data:
            return
        try:
            self._write_time_series(data)
        except Exception as exc:
            # Never let a monitoring failure affect the serving path.
            logger.warning("Cloud Monitoring flush failed (will retry): %s", exc)

    def _write_time_series(self, data: Dict[Tuple[str, str], _Bucket]) -> None:
        from google.cloud import monitoring_v3

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{self._project_id}"
        now_seconds = int(time.time())

        series_list = []
        for (precision, endpoint), b in data.items():
            labels = {"model_precision": precision, "endpoint": endpoint}
            avg_latency = b.latency_sum / b.count if b.count else 0.0
            avg_inference = b.inference_sum / b.count if b.count else 0.0

            series_list.extend([
                self._make_series(
                    f"{_METRIC_PREFIX}/request_count",
                    labels, b.count, now_seconds, value_type="int",
                ),
                self._make_series(
                    f"{_METRIC_PREFIX}/error_count",
                    labels, b.errors, now_seconds, value_type="int",
                ),
                self._make_series(
                    f"{_METRIC_PREFIX}/request_latency_ms",
                    labels, avg_latency, now_seconds, value_type="double",
                ),
                self._make_series(
                    f"{_METRIC_PREFIX}/inference_latency_ms",
                    labels, avg_inference, now_seconds, value_type="double",
                ),
            ])

        if series_list:
            client.create_time_series(name=project_name, time_series=series_list)
            logger.debug(
                "Flushed %d time series to Cloud Monitoring.", len(series_list)
            )

    def _make_series(self, metric_type: str, labels: dict,
                     value, now_seconds: int, value_type: str):
        from google.cloud import monitoring_v3

        series = monitoring_v3.TimeSeries()
        series.metric.type = metric_type
        for k, v in labels.items():
            series.metric.labels[k] = str(v)
        series.resource.type = "global"
        series.resource.labels["project_id"] = self._project_id

        point = monitoring_v3.Point()
        point.interval.end_time.seconds = now_seconds
        if value_type == "int":
            point.value.int64_value = int(value)
        else:
            point.value.double_value = float(value)

        series.points = [point]
        return series

    @staticmethod
    def _fetch_project_from_metadata() -> str:
        """Try the GCE/Cloud Run metadata server; returns '' on failure."""
        import urllib.request
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
            )
            return urllib.request.urlopen(req, timeout=2).read().decode()
        except Exception:
            return ""


# ── Module-level singleton ─────────────────────────────────────────────────────
# Imported and lifecycle-managed by api/main.py.

reporter = CloudMonitoringReporter(
    flush_interval=int(os.environ.get("CM_FLUSH_INTERVAL", "60"))
)
