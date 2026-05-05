"""
Adaptive Budget Scheduler for KV-Cache Compression.

As context grows, a fixed compression budget becomes increasingly aggressive
(or conversely, too generous). This module provides schedulers that
*dynamically adjust* the compression budget based on the current sequence
length and available memory.

Schedulers
----------
LinearBudgetScheduler     — linearly ramps compression from mild to aggressive
ExponentialBudgetScheduler — exponential ramp (aggressive early, milder later)
StepBudgetScheduler       — discrete step function (thresholds)
MemoryAwareBudgetScheduler — adjusts budget based on remaining GPU memory

All schedulers expose a unified API:
    budget = scheduler.get_budget(seq_len)  → CompressionBudget

The returned CompressionBudget can be fed directly into any compression policy.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Budget dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompressionBudget:
    """
    A resolved compression budget for a specific sequence length.

    Attributes
    ----------
    keep_last_tokens : recency window size
    keep_top_tokens  : attention-top-k budget
    cluster_tokens   : number of cluster representatives (0 = no clustering)
    num_sink_tokens  : number of sink tokens (for StreamingLLM style)
    compression_ratio: expected ratio of kept/original tokens (informational)
    """
    keep_last_tokens: int
    keep_top_tokens: int
    cluster_tokens: int = 0
    num_sink_tokens: int = 4
    compression_ratio: float = 1.0

    @property
    def total_budget(self) -> int:
        return self.num_sink_tokens + self.keep_last_tokens + self.keep_top_tokens + self.cluster_tokens

    def __repr__(self) -> str:
        return (
            f"CompressionBudget("
            f"sink={self.num_sink_tokens}, last={self.keep_last_tokens}, "
            f"top={self.keep_top_tokens}, clusters={self.cluster_tokens}, "
            f"ratio={self.compression_ratio:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BudgetScheduler(ABC):
    """Abstract base class for all budget schedulers."""

    @abstractmethod
    def get_budget(self, seq_len: int) -> CompressionBudget:
        """Return the compression budget for the given sequence length."""
        ...

    def describe(self) -> str:
        return f"{self.__class__.__name__}"


# ──────────────────────────────────────────────────────────────────────────────
# 1. LinearBudgetScheduler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LinearBudgetScheduler(BudgetScheduler):
    """
    Linearly interpolates between a mild budget (short context) and an
    aggressive budget (long context).

    Example
    -------
    At seq_len = start_len  → compression_ratio = start_ratio  (e.g. 0.9 = mild)
    At seq_len = end_len    → compression_ratio = end_ratio    (e.g. 0.3 = aggressive)
    Beyond end_len          → clamped at end_ratio

    Parameters
    ----------
    start_len        : sequence length where scheduling begins
    end_len          : sequence length at maximum compression
    start_ratio      : fraction of tokens to keep at start_len
    end_ratio        : fraction of tokens to keep at end_len
    num_sink_tokens  : always-fixed sink token count
    min_last_tokens  : minimum recency window (safety floor)
    """
    start_len: int = 512
    end_len: int = 4096
    start_ratio: float = 0.9
    end_ratio: float = 0.25
    num_sink_tokens: int = 4
    min_last_tokens: int = 64

    def get_budget(self, seq_len: int) -> CompressionBudget:
        if seq_len <= self.start_len:
            ratio = self.start_ratio
        elif seq_len >= self.end_len:
            ratio = self.end_ratio
        else:
            t = (seq_len - self.start_len) / (self.end_len - self.start_len)
            ratio = self.start_ratio + t * (self.end_ratio - self.start_ratio)

        total_keep = max(int(seq_len * ratio), self.num_sink_tokens + self.min_last_tokens)
        non_sink = total_keep - self.num_sink_tokens
        keep_last = max(self.min_last_tokens, non_sink // 2)
        keep_top  = max(non_sink - keep_last, 0)

        return CompressionBudget(
            keep_last_tokens=keep_last,
            keep_top_tokens=keep_top,
            cluster_tokens=0,
            num_sink_tokens=self.num_sink_tokens,
            compression_ratio=ratio,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. ExponentialBudgetScheduler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExponentialBudgetScheduler(BudgetScheduler):
    """
    Exponentially decaying budget ratio: more aggressive as context grows.

    ratio(L) = end_ratio + (start_ratio - end_ratio) * exp(-lambda * (L - start_len) / scale)

    This models the intuition that the marginal value of an extra token
    decreases as the context grows (diminishing returns).

    Parameters
    ----------
    start_len    : length below which no compression is applied
    scale        : decay scale in tokens
    start_ratio  : ratio at start_len
    end_ratio    : asymptotic minimum ratio
    decay_lambda : decay rate (larger = faster decay)
    """
    start_len: int = 512
    scale: int = 2048
    start_ratio: float = 0.85
    end_ratio: float = 0.20
    decay_lambda: float = 1.5
    num_sink_tokens: int = 4
    min_last_tokens: int = 64

    def get_budget(self, seq_len: int) -> CompressionBudget:
        if seq_len <= self.start_len:
            ratio = self.start_ratio
        else:
            t = (seq_len - self.start_len) / self.scale
            ratio = self.end_ratio + (self.start_ratio - self.end_ratio) * math.exp(
                -self.decay_lambda * t
            )
            ratio = max(ratio, self.end_ratio)

        total_keep = max(int(seq_len * ratio), self.num_sink_tokens + self.min_last_tokens)
        non_sink = total_keep - self.num_sink_tokens
        keep_last = max(self.min_last_tokens, non_sink // 2)
        keep_top  = max(non_sink - keep_last, 0)

        return CompressionBudget(
            keep_last_tokens=keep_last,
            keep_top_tokens=keep_top,
            cluster_tokens=0,
            num_sink_tokens=self.num_sink_tokens,
            compression_ratio=ratio,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. StepBudgetScheduler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepBudgetScheduler(BudgetScheduler):
    """
    Discrete step-function budget: applies fixed budgets within length ranges.

    Useful when you want explicit, predictable control over compression at
    specific context length thresholds.

    Parameters
    ----------
    steps : list of (max_seq_len, CompressionBudget) tuples, sorted ascending.
            The last entry's max_seq_len should be very large (or math.inf).
            When seq_len <= max_seq_len, that step's budget is used.

    Example
    -------
    scheduler = StepBudgetScheduler(steps=[
        (512,  CompressionBudget(keep_last_tokens=512,  keep_top_tokens=0)),
        (2048, CompressionBudget(keep_last_tokens=256, keep_top_tokens=128)),
        (8192, CompressionBudget(keep_last_tokens=128, keep_top_tokens=64, cluster_tokens=32)),
    ])
    """
    steps: list[tuple[int, CompressionBudget]]

    def __post_init__(self) -> None:
        self.steps = sorted(self.steps, key=lambda x: x[0])

    def get_budget(self, seq_len: int) -> CompressionBudget:
        for max_len, budget in self.steps:
            if seq_len <= max_len:
                return budget
        # Beyond all steps: use last (most aggressive)
        return self.steps[-1][1]


# ──────────────────────────────────────────────────────────────────────────────
# 4. MemoryAwareBudgetScheduler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryAwareBudgetScheduler(BudgetScheduler):
    """
    Adjusts compression budget dynamically based on available GPU memory.

    When GPU memory is plentiful → mild compression (high ratio).
    When GPU memory is running low → aggressive compression (low ratio).

    Falls back to ExponentialBudgetScheduler if GPU is not available.

    Parameters
    ----------
    target_free_bytes : desired free GPU memory to maintain (bytes)
    max_ratio         : ratio when memory is ample
    min_ratio         : ratio when memory is critically low
    num_sink_tokens   : always-fixed sink count
    min_last_tokens   : safety floor for recency window
    """
    target_free_bytes: int = 2 * 1024 ** 3   # 2 GB default
    max_ratio: float = 0.9
    min_ratio: float = 0.15
    num_sink_tokens: int = 4
    min_last_tokens: int = 64
    _fallback: BudgetScheduler | None = None

    def __post_init__(self) -> None:
        self._fallback = ExponentialBudgetScheduler(
            num_sink_tokens=self.num_sink_tokens,
            min_last_tokens=self.min_last_tokens,
        )

    def _gpu_free_bytes(self) -> int | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free, _ = torch.cuda.mem_get_info()
            return int(free)
        except Exception:
            return None

    def get_budget(self, seq_len: int) -> CompressionBudget:
        free = self._gpu_free_bytes()
        if free is None:
            # No GPU or cannot query → fall back to exponential scheduler
            return self._fallback.get_budget(seq_len)

        # Interpolate ratio based on how much free memory remains
        # free >= 2*target → max_ratio (ample memory)
        # free <= 0        → min_ratio (critical)
        t = max(0.0, min(1.0, free / (2 * self.target_free_bytes)))
        ratio = self.min_ratio + t * (self.max_ratio - self.min_ratio)

        total_keep = max(int(seq_len * ratio), self.num_sink_tokens + self.min_last_tokens)
        non_sink = total_keep - self.num_sink_tokens
        keep_last = max(self.min_last_tokens, non_sink // 2)
        keep_top  = max(non_sink - keep_last, 0)

        return CompressionBudget(
            keep_last_tokens=keep_last,
            keep_top_tokens=keep_top,
            cluster_tokens=0,
            num_sink_tokens=self.num_sink_tokens,
            compression_ratio=ratio,
        )

    def describe(self) -> str:
        free = self._gpu_free_bytes()
        free_str = f"{free / 1024**3:.2f} GB" if free is not None else "N/A (CPU)"
        return f"MemoryAwareBudgetScheduler(free_gpu={free_str}, target={self.target_free_bytes/1024**3:.1f} GB)"


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

def make_scheduler(
    name: str,
    *,
    start_len: int = 512,
    end_len: int = 4096,
    start_ratio: float = 0.9,
    end_ratio: float = 0.25,
    num_sink_tokens: int = 4,
    min_last_tokens: int = 64,
) -> BudgetScheduler:
    """
    Factory: create a named scheduler with common parameters.

    name choices: "linear", "exponential", "memory"
    """
    if name == "linear":
        return LinearBudgetScheduler(
            start_len=start_len, end_len=end_len,
            start_ratio=start_ratio, end_ratio=end_ratio,
            num_sink_tokens=num_sink_tokens, min_last_tokens=min_last_tokens,
        )
    if name == "exponential":
        return ExponentialBudgetScheduler(
            start_len=start_len, scale=end_len,
            start_ratio=start_ratio, end_ratio=end_ratio,
            num_sink_tokens=num_sink_tokens, min_last_tokens=min_last_tokens,
        )
    if name == "memory":
        return MemoryAwareBudgetScheduler(
            num_sink_tokens=num_sink_tokens, min_last_tokens=min_last_tokens,
        )
    raise ValueError(f"Unknown scheduler name '{name}'. Choose from: linear, exponential, memory")
