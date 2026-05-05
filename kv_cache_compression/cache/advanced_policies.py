"""
Advanced KV-cache compression policies.

Implements research-grade strategies beyond simple recency/attention:

1. SinkTokenPolicy     — StreamingLLM-style: always keep first N "sink" tokens
                         (they act as attention sinks and are crucial for stability)
                         combined with a sliding recency window.

2. HeavyHitterPolicy   — H2O (Heavy Hitter Oracle) style: track cumulative
                         attention scores across all forward passes; evict tokens
                         that have the lowest *accumulated* attention mass.

3. QuantizedCachePolicy — Reduces memory without eviction by quantizing key/value
                          tensors to int8 (per-channel dynamic quantization).
                          No token is dropped; instead the representation precision
                          is reduced, saving ~50% memory.

4. ScissorHandsPolicy  — Combines sink tokens + heavy hitters + recency in a
                         unified budget allocation.

References:
  - StreamingLLM : Xiao et al. (2023), "Efficient Streaming Language Models with
                   Attention Sinks", arXiv:2309.17453
  - H2O           : Zhang et al. (2023), "H2O: Heavy-Hitter Oracle for Efficient
                   Generative Inference of Large Language Models", arXiv:2306.14048
  - ScissorHands  : Liu et al. (2023), "ScissorHands: Exploiting the Persistence
                   of Importance Hypothesis for LLM KV Cache Compression",
                   arXiv:2305.17118
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .kv_cache import (
    CompressionOutcome,
    KVCacheInspector,
    KVCacheTensor,
    select_token_indices,
)
from .policies import CompressionPolicy, PolicyContext
from .prune import build_recency_indices


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SinkTokenPolicy  (StreamingLLM)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SinkTokenPolicy(CompressionPolicy):
    """
    StreamingLLM-style attention-sink + recency window compression.

    Key insight (Xiao et al. 2023):
      Transformer attention is NOT uniform — the very first few tokens (position 0–3)
      accumulate disproportionate attention mass regardless of their semantic content.
      They act as "sinks". Evicting them destroys model stability even when they
      carry no useful information.

    Strategy:
      - Always keep the first `num_sink_tokens` positions (sinks).
      - Keep the last `keep_last_tokens` positions (recency window).
      - Discard everything in between.

    This enables theoretically *infinite* streaming context at fixed cache size
    (num_sink_tokens + keep_last_tokens).

    Parameters
    ----------
    num_sink_tokens  : number of initial tokens to always preserve (typically 1–4)
    keep_last_tokens : size of sliding recency window
    """
    num_sink_tokens: int
    keep_last_tokens: int
    name: str = "sink_token"

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)

        # Sink indices: always first num_sink_tokens
        sink_indices = list(range(min(self.num_sink_tokens, original_tokens)))

        # Recency indices: last keep_last_tokens (non-overlapping with sinks)
        recency_start = max(original_tokens - self.keep_last_tokens, self.num_sink_tokens)
        recency_indices = list(range(recency_start, original_tokens))

        # Union, sorted
        kept_indices = sorted(set(sink_indices) | set(recency_indices))

        compressed = select_token_indices(past_key_values, kept_indices)
        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed,
            original_tokens=original_tokens,
            compressed_tokens=len(kept_indices),
            kept_token_indices=kept_indices,
            metadata={
                "num_sink_tokens": self.num_sink_tokens,
                "keep_last_tokens": self.keep_last_tokens,
                "sink_indices": sink_indices,
                "recency_start": recency_start,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HeavyHitterPolicy  (H2O)
# ══════════════════════════════════════════════════════════════════════════════

class HeavyHitterOracle:
    """
    Stateful accumulator for cumulative per-token attention scores.

    H2O insight (Zhang et al. 2023):
      A small subset of tokens consistently attract the majority of attention
      mass across *multiple* forward passes. These "heavy hitters" should be
      preserved in the KV cache while low-attention tokens are evicted.

    Usage:
        oracle = HeavyHitterOracle(budget=512)
        oracle.update(attention_scores)   # call after every forward pass
        kept = oracle.top_indices(budget=256)
    """

    def __init__(self, decay: float = 0.9) -> None:
        """
        Parameters
        ----------
        decay : exponential decay factor for accumulated scores (0 < decay ≤ 1).
                decay=1.0 → pure sum; decay<1 → recent attention weighted higher.
        """
        self.decay = decay
        self._accumulated: torch.Tensor | None = None

    def update(self, attention_scores: torch.Tensor) -> None:
        """
        Update accumulated scores.

        Parameters
        ----------
        attention_scores : [1, seq_len] tensor of per-token attention scores
        """
        if attention_scores is None:
            return
        scores = attention_scores[0].detach().float()

        if self._accumulated is None or self._accumulated.shape[0] != scores.shape[0]:
            # Initialise or reset on shape change (new prompt)
            self._accumulated = scores.clone()
        else:
            self._accumulated = self.decay * self._accumulated + scores

    def top_indices(self, budget: int, seq_len: int) -> list[int]:
        """Return indices of the top-`budget` heavy hitters."""
        if self._accumulated is None:
            return build_recency_indices(seq_len, budget)
        k = min(budget, self._accumulated.shape[0])
        return torch.topk(self._accumulated, k=k).indices.sort().values.tolist()

    def reset(self) -> None:
        self._accumulated = None


@dataclass
class HeavyHitterPolicy(CompressionPolicy):
    """
    H2O-style policy: keep tokens with highest *cumulative* attention mass.

    Maintains a HeavyHitterOracle internally; call `update_oracle` with each
    new attention tensor before calling `compress`.

    Parameters
    ----------
    budget           : total KV cache budget (max tokens to keep)
    num_sink_tokens  : always keep these initial sink tokens
    decay            : exponential decay for accumulated scores
    """
    budget: int
    num_sink_tokens: int = 4
    decay: float = 0.9
    name: str = "heavy_hitter"

    def __post_init__(self) -> None:
        self.oracle: HeavyHitterOracle = HeavyHitterOracle(decay=self.decay)

    def update_oracle(self, attention_scores: torch.Tensor) -> None:
        """Feed latest attention scores into the accumulator."""
        self.oracle.update(attention_scores)

    def reset(self) -> None:
        """Reset accumulated state (call between independent documents)."""
        self.oracle.reset()

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)

        # Update oracle with latest attention if available
        if context is not None and context.attention_scores is not None:
            self.oracle.update(context.attention_scores)

        # Always keep sink tokens
        sink_indices = set(range(min(self.num_sink_tokens, original_tokens)))

        # Fill remaining budget with heavy hitters
        remaining_budget = max(self.budget - len(sink_indices), 0)
        hitter_indices = set(self.oracle.top_indices(remaining_budget, original_tokens))

        kept_indices = sorted(sink_indices | hitter_indices)

        compressed = select_token_indices(past_key_values, kept_indices)
        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed,
            original_tokens=original_tokens,
            compressed_tokens=len(kept_indices),
            kept_token_indices=kept_indices,
            metadata={
                "budget": self.budget,
                "num_sink_tokens": self.num_sink_tokens,
                "decay": self.decay,
                "oracle_initialized": self.oracle._accumulated is not None,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  QuantizedCachePolicy  (INT8 KV Quantization)
# ══════════════════════════════════════════════════════════════════════════════

def _quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-token dynamic INT8 quantization.

    Quantizes along the last dimension (head_dim) per (batch, head, token).
    Returns (quantized_int8, scale, zero_point).
    """
    orig_shape = tensor.shape
    # Flatten to [..., head_dim]
    flat = tensor.reshape(-1, orig_shape[-1]).float()  # [N, head_dim]

    min_val = flat.min(dim=-1, keepdim=True).values
    max_val = flat.max(dim=-1, keepdim=True).values

    scale = (max_val - min_val) / 255.0
    scale = scale.clamp(min=1e-8)
    zero_point = (-128 - min_val / scale).round().clamp(-128, 127)

    quantized = (flat / scale + zero_point).round().clamp(-128, 127).to(torch.int8)

    return (
        quantized.reshape(orig_shape[:-1] + (orig_shape[-1],)),
        scale.reshape(orig_shape[:-1] + (1,)),
        zero_point.reshape(orig_shape[:-1] + (1,)),
    )


def _dequantize_tensor_int8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct a float tensor from INT8 quantized representation."""
    return ((quantized.float() - zero_point) * scale).to(target_dtype)


@dataclass
class QuantizedKVLayer:
    """Stores one layer's quantized key and value tensors."""
    key_q: torch.Tensor      # int8
    key_scale: torch.Tensor
    key_zero: torch.Tensor
    val_q: torch.Tensor      # int8
    val_scale: torch.Tensor
    val_zero: torch.Tensor
    orig_dtype: torch.dtype

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        key = _dequantize_tensor_int8(self.key_q, self.key_scale, self.key_zero, self.orig_dtype)
        val = _dequantize_tensor_int8(self.val_q, self.val_scale, self.val_zero, self.orig_dtype)
        return key, val

    def memory_bytes(self) -> int:
        def _b(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()
        return sum(_b(t) for t in [
            self.key_q, self.key_scale, self.key_zero,
            self.val_q, self.val_scale, self.val_zero,
        ])


@dataclass
class QuantizedCachePolicy(CompressionPolicy):
    """
    INT8 quantization of KV tensors — no token eviction, pure precision reduction.

    Memory saving: ~50% vs float16 (int8 = 1 byte vs 2 bytes per element).

    The compressed cache returned uses dequantized tensors (float16/bfloat16)
    so downstream model inference is unaffected. The `quantized_layers` attribute
    stores the raw INT8 data if you want to measure actual compressed size.

    Parameters
    ----------
    keep_last_tokens : if > 0, recent tokens are kept at full precision;
                       older tokens are quantized. 0 = quantize everything.
    """
    keep_last_tokens: int = 0
    name: str = "quantized_int8"

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)
        original_bytes = KVCacheInspector.total_bytes(past_key_values)

        quantized_layers: list[QuantizedKVLayer] = []
        reconstructed_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        quantized_bytes = 0

        for key, value in past_key_values:
            orig_dtype = key.dtype

            if self.keep_last_tokens > 0 and original_tokens > self.keep_last_tokens:
                split = original_tokens - self.keep_last_tokens

                # Quantize the "old" portion
                key_old, key_new = key[..., :split, :], key[..., split:, :]
                val_old, val_new = value[..., :split, :], value[..., split:, :]

                key_q, key_scale, key_zero = _quantize_tensor_int8(key_old)
                val_q, val_scale, val_zero = _quantize_tensor_int8(val_old)

                ql = QuantizedKVLayer(
                    key_q=key_q, key_scale=key_scale, key_zero=key_zero,
                    val_q=val_q, val_scale=val_scale, val_zero=val_zero,
                    orig_dtype=orig_dtype,
                )
                quantized_layers.append(ql)
                quantized_bytes += ql.memory_bytes()

                # Reconstruct and concat
                key_deq, val_deq = ql.dequantize()
                reconstructed_layers.append((
                    torch.cat([key_deq, key_new], dim=-2),
                    torch.cat([val_deq, val_new], dim=-2),
                ))
            else:
                # Quantize everything
                key_q, key_scale, key_zero = _quantize_tensor_int8(key)
                val_q, val_scale, val_zero = _quantize_tensor_int8(value)

                ql = QuantizedKVLayer(
                    key_q=key_q, key_scale=key_scale, key_zero=key_zero,
                    val_q=val_q, val_scale=val_scale, val_zero=val_zero,
                    orig_dtype=orig_dtype,
                )
                quantized_layers.append(ql)
                quantized_bytes += ql.memory_bytes()

                key_deq, val_deq = ql.dequantize()
                reconstructed_layers.append((key_deq, val_deq))

        compressed_cache = tuple(reconstructed_layers)

        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed_cache,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,   # no token eviction
            kept_token_indices=list(range(original_tokens)),
            metadata={
                "keep_last_tokens": self.keep_last_tokens,
                "original_bytes": original_bytes,
                "quantized_bytes": quantized_bytes,
                "memory_reduction_pct": round(
                    (1 - quantized_bytes / max(original_bytes, 1)) * 100, 2
                ),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ScissorHandsPolicy  (Sink + Heavy-Hitter + Recency unified)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScissorHandsPolicy(CompressionPolicy):
    """
    Unified policy combining sink tokens, attention-based heavy hitters,
    and a recency window — three non-overlapping budget slots.

    Inspiration: ScissorHands (Liu et al. 2023) — persistent importance
    hypothesis: tokens important at step T remain important at step T+k.

    Budget split:
        total_budget = num_sink_tokens + keep_top_tokens + keep_last_tokens

    Parameters
    ----------
    num_sink_tokens  : always-kept initial tokens (attention sinks)
    keep_top_tokens  : kept by cumulative attention importance (heavy hitters)
    keep_last_tokens : kept by recency window
    decay            : exponential decay for accumulated attention scores
    """
    num_sink_tokens: int
    keep_top_tokens: int
    keep_last_tokens: int
    decay: float = 0.9
    name: str = "scissorhands"

    def __post_init__(self) -> None:
        self._oracle: HeavyHitterOracle = HeavyHitterOracle(decay=self.decay)

    def reset(self) -> None:
        self._oracle.reset()

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)

        # Update oracle
        if context is not None and context.attention_scores is not None:
            self._oracle.update(context.attention_scores)

        # Slot 1: sink tokens (always first N)
        sink_set = set(range(min(self.num_sink_tokens, original_tokens)))

        # Slot 2: recency window (last M tokens, excluding sinks)
        recency_start = max(original_tokens - self.keep_last_tokens, self.num_sink_tokens)
        recency_set = set(range(recency_start, original_tokens))

        # Slot 3: heavy hitters from middle region (excluding sink & recency)
        middle_candidates = [
            i for i in range(self.num_sink_tokens, recency_start)
        ]
        if middle_candidates and self.keep_top_tokens > 0 and self._oracle._accumulated is not None:
            acc = self._oracle._accumulated
            if acc.shape[0] >= len(middle_candidates) + self.num_sink_tokens:
                middle_scores = acc[self.num_sink_tokens:recency_start]
                k = min(self.keep_top_tokens, len(middle_candidates))
                top_local = torch.topk(middle_scores, k=k).indices
                hitter_set = {self.num_sink_tokens + int(i) for i in top_local.tolist()}
            else:
                hitter_set = set()
        else:
            hitter_set = set()

        kept_indices = sorted(sink_set | recency_set | hitter_set)
        compressed = select_token_indices(past_key_values, kept_indices)

        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed,
            original_tokens=original_tokens,
            compressed_tokens=len(kept_indices),
            kept_token_indices=kept_indices,
            metadata={
                "num_sink_tokens": self.num_sink_tokens,
                "keep_top_tokens": self.keep_top_tokens,
                "keep_last_tokens": self.keep_last_tokens,
                "decay": self.decay,
                "n_sink": len(sink_set),
                "n_recency": len(recency_set),
                "n_hitters": len(hitter_set),
                "total_budget": self.num_sink_tokens + self.keep_top_tokens + self.keep_last_tokens,
            },
        )
