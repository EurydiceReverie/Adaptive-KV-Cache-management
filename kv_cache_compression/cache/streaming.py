"""
Streaming KV-Cache Compression Manager.

Enables continuous / auto-regressive generation with bounded KV-cache size
by applying compression periodically as new tokens are generated.

Key concepts
------------
- The model generates one token at a time, appending to past_key_values.
- Without compression, memory grows linearly with context length.
- With StreamingCompressor, compression fires every `compress_every` tokens.
- An optional BudgetScheduler dynamically adjusts the budget each time.

Usage (pseudo-code)
-------------------
    manager = StreamingCompressor(
        policy=SinkTokenPolicy(num_sink_tokens=4, keep_last_tokens=512),
        compress_every=64,
        scheduler=LinearBudgetScheduler(),
    )

    for step in range(max_new_tokens):
        out = model(input_ids=next_token, past_key_values=current_kv, use_cache=True)
        current_kv = to_legacy_tuple(out.past_key_values)
        current_kv, stats = manager.step(current_kv, attention_scores=out.attentions)

Classes
-------
StreamingStats          — per-step statistics
StreamingCompressor     — main stateful manager
StreamingGenerationMixin— mixin to add streaming compression to any generator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .kv_cache import KVCacheInspector, KVCacheTensor, to_legacy_tuple
from .policies import CompressionPolicy, PolicyContext
from .prune import aggregate_attention_scores
from .scheduler import BudgetScheduler, CompressionBudget


# ──────────────────────────────────────────────────────────────────────────────
# Stats dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamingStats:
    """Statistics from one streaming compression step."""
    step: int
    seq_len_before: int
    seq_len_after: int
    compressed: bool
    tokens_evicted: int
    compression_ratio: float
    budget: CompressionBudget | None = None

    def __repr__(self) -> str:
        if self.compressed:
            return (
                f"StreamingStats(step={self.step}, "
                f"{self.seq_len_before}→{self.seq_len_after} tokens, "
                f"evicted={self.tokens_evicted}, ratio={self.compression_ratio:.3f})"
            )
        return f"StreamingStats(step={self.step}, no_compression, len={self.seq_len_before})"


# ──────────────────────────────────────────────────────────────────────────────
# Core: StreamingCompressor
# ──────────────────────────────────────────────────────────────────────────────

class StreamingCompressor:
    """
    Stateful streaming KV-cache compression manager.

    Wraps any CompressionPolicy and applies it every `compress_every` tokens.
    Optionally integrates with a BudgetScheduler to adapt budgets dynamically.

    Parameters
    ----------
    policy          : any CompressionPolicy instance
    compress_every  : compress after every N new tokens (default: 64)
    scheduler       : optional BudgetScheduler — if provided, the policy's
                      budget params are overridden per step
    warmup_tokens   : do not compress until at least this many tokens are present
    on_compress     : optional callback(stats: StreamingStats) called after each compression
    """

    def __init__(
        self,
        policy: CompressionPolicy,
        compress_every: int = 64,
        scheduler: BudgetScheduler | None = None,
        warmup_tokens: int = 128,
        on_compress: Callable[[StreamingStats], None] | None = None,
    ) -> None:
        self.policy = policy
        self.compress_every = compress_every
        self.scheduler = scheduler
        self.warmup_tokens = warmup_tokens
        self.on_compress = on_compress

        self._step: int = 0
        self._tokens_since_compress: int = 0
        self._history: list[StreamingStats] = []

    @property
    def step(self) -> int:
        return self._step

    @property
    def history(self) -> list[StreamingStats]:
        return list(self._history)

    def _apply_scheduler_budget(self, seq_len: int) -> None:
        """Override policy budget params using scheduler if available."""
        if self.scheduler is None:
            return
        budget = self.scheduler.get_budget(seq_len)
        # Patch policy params if they exist (duck-typed)
        for attr, val in [
            ("keep_last_tokens", budget.keep_last_tokens),
            ("keep_top_tokens", budget.keep_top_tokens),
            ("num_sink_tokens", budget.num_sink_tokens),
            ("cluster_tokens", budget.cluster_tokens),
        ]:
            if hasattr(self.policy, attr):
                object.__setattr__(self.policy, attr, val) if hasattr(
                    self.policy, "__slots__"
                ) else setattr(self.policy, attr, val)

    def step_cache(
        self,
        past_key_values: KVCacheTensor,
        attention_scores: torch.Tensor | None = None,
        attentions_tuple: tuple | None = None,
        special_token_mask: torch.Tensor | None = None,
        force: bool = False,
    ) -> tuple[KVCacheTensor, StreamingStats]:
        """
        Process one generation step.

        Call this after every model forward pass during generation.

        Parameters
        ----------
        past_key_values     : current KV cache (legacy tuple format)
        attention_scores    : [1, seq_len] tensor of token importance scores
        attentions_tuple    : raw model attentions tuple (auto-aggregated if provided)
        special_token_mask  : [1, seq_len] bool mask for special tokens
        force               : if True, compress regardless of step counter

        Returns
        -------
        (compressed_kv, stats)
        """
        self._step += 1
        seq_len_before = KVCacheInspector.sequence_length(past_key_values)
        self._tokens_since_compress += 1

        should_compress = (
            force
            or (
                self._tokens_since_compress >= self.compress_every
                and seq_len_before >= self.warmup_tokens
            )
        )

        if not should_compress:
            stats = StreamingStats(
                step=self._step,
                seq_len_before=seq_len_before,
                seq_len_after=seq_len_before,
                compressed=False,
                tokens_evicted=0,
                compression_ratio=1.0,
            )
            self._history.append(stats)
            return past_key_values, stats

        # Resolve attention scores
        if attentions_tuple is not None and attention_scores is None:
            attention_scores = aggregate_attention_scores(attentions_tuple)

        # Apply scheduler budget adjustment
        if self.scheduler is not None:
            budget = self.scheduler.get_budget(seq_len_before)
            self._apply_scheduler_budget(seq_len_before)
        else:
            budget = None

        context = PolicyContext(
            attention_scores=attention_scores,
            special_token_mask=special_token_mask,
        )

        try:
            outcome = self.policy.compress(past_key_values, context=context)
            compressed_kv = outcome.past_key_values
            seq_len_after = KVCacheInspector.sequence_length(compressed_kv)
        except Exception as e:
            # Graceful degradation: if compression fails, return original
            print(f"[StreamingCompressor] Warning: compression failed at step {self._step}: {e}")
            compressed_kv = past_key_values
            seq_len_after = seq_len_before

        self._tokens_since_compress = 0

        stats = StreamingStats(
            step=self._step,
            seq_len_before=seq_len_before,
            seq_len_after=seq_len_after,
            compressed=True,
            tokens_evicted=seq_len_before - seq_len_after,
            compression_ratio=seq_len_after / max(seq_len_before, 1),
            budget=budget,
        )
        self._history.append(stats)
        if self.on_compress is not None:
            self.on_compress(stats)

        return compressed_kv, stats

    def reset(self) -> None:
        """Reset step counter and history (call between independent documents)."""
        self._step = 0
        self._tokens_since_compress = 0
        self._history.clear()
        # Reset stateful policies (H2O oracle, ScissorHands, etc.)
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def summary(self) -> dict:
        """Return a summary of compression statistics over all steps."""
        compressions = [s for s in self._history if s.compressed]
        if not compressions:
            return {"total_steps": self._step, "compressions": 0}
        total_evicted = sum(s.tokens_evicted for s in compressions)
        avg_ratio = sum(s.compression_ratio for s in compressions) / len(compressions)
        return {
            "total_steps": self._step,
            "compressions": len(compressions),
            "total_tokens_evicted": total_evicted,
            "avg_compression_ratio": round(avg_ratio, 4),
            "compress_every": self.compress_every,
            "policy": self.policy.name,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: run a full generation loop with streaming compression
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_with_compression(
    model,
    tokenizer,
    prompt: str,
    compressor: StreamingCompressor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str | torch.device = "cpu",
) -> tuple[str, dict]:
    """
    Run greedy/nucleus generation with streaming KV-cache compression.

    Returns
    -------
    (generated_text, compression_summary)
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    compressor.reset()

    # Prefill
    outputs = model(input_ids=input_ids, use_cache=True, output_attentions=False)
    past_kv = to_legacy_tuple(outputs.past_key_values)
    logits = outputs.logits[:, -1, :]

    generated_ids = []

    for _ in range(max_new_tokens):
        # Sample next token
        if temperature == 0.0 or temperature < 1e-6:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cum_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = cum_probs - sorted_probs > top_p
                sorted_probs[remove_mask] = 0.0
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                probs /= probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids.append(int(next_token.item()))

        # Check EOS
        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            break

        # Step the model
        out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv = to_legacy_tuple(out.past_key_values)
        logits = out.logits[:, -1, :]

        # Streaming compression step
        past_kv, _ = compressor.step_cache(
            past_kv,
            attention_scores=None,   # key-norm fallback inside policy
        )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, compressor.summary()
