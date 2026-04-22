"""
profiling.py – PyTorch Profiler integration for per-request latency breakdown.

Breaks each request into three stages:
  1. preprocess  – image decode + tensor transforms
  2. forward     – model forward pass
  3. postprocess – softmax + top-k extraction + serialization

Profiling is opt-in via the `return_profile=true` query param and adds ~5-10 ms overhead.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class StageTimer:
    preprocess_ms:  float = 0.0
    forward_ms:     float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.forward_ms + self.postprocess_ms

    def to_dict(self) -> dict:
        return {
            "preprocess_ms":  round(self.preprocess_ms, 3),
            "forward_ms":     round(self.forward_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "total_ms":       round(self.total_ms, 3),
        }


@contextmanager
def timer():
    """Simple wall-clock context manager. Returns elapsed_ms via a list."""
    t = [0.0]
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t[0] = (time.perf_counter() - t0) * 1000


class RequestProfiler:
    """
    Context manager that wraps a full inference request with PyTorch Profiler
    when `enabled=True`, and falls back to simple wall-clock timers otherwise.

    Usage:
        profiler = RequestProfiler(enabled=request.return_profile)
        with profiler:
            with profiler.stage("preprocess"):
                tensor = preprocess(image)
            with profiler.stage("forward"):
                logits = model(tensor)
            with profiler.stage("postprocess"):
                result = postprocess(logits)
        breakdown = profiler.breakdown
    """

    def __init__(self, enabled: bool = False, use_cuda: bool = False):
        self.enabled  = enabled
        self.use_cuda = use_cuda
        self._timers: StageTimer = StageTimer()
        self._torch_profiler: Optional[torch.profiler.profile] = None
        self._stage_start: float = 0.0
        self._current_stage: Optional[str] = None

    def __enter__(self):
        if self.enabled:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.use_cuda and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._torch_profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            )
            self._torch_profiler.__enter__()
        return self

    def __exit__(self, *args):
        if self._torch_profiler is not None:
            self._torch_profiler.__exit__(*args)

    @contextmanager
    def stage(self, name: str):
        """Time a named stage."""
        t0 = time.perf_counter()
        try:
            if self._torch_profiler:
                with torch.profiler.record_function(name):
                    yield
            else:
                yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            if name == "preprocess":
                self._timers.preprocess_ms = elapsed
            elif name == "forward":
                self._timers.forward_ms = elapsed
            elif name == "postprocess":
                self._timers.postprocess_ms = elapsed

    @property
    def breakdown(self) -> StageTimer:
        return self._timers

    def torch_summary(self, top: int = 10) -> str:
        """Return a formatted PyTorch Profiler table (requires enabled=True)."""
        if self._torch_profiler is None:
            return "Profiler not enabled."
        return self._torch_profiler.key_averages().table(
            sort_by="cpu_time_total", row_limit=top
        )
