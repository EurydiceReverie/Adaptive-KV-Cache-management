from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_cache import CompressionOutcome, KVCacheInspector, select_token_indices
from .policies import CompressionPolicy, PolicyContext


def aggregate_attention_scores(attentions: tuple[torch.Tensor, ...] | None) -> torch.Tensor | None:
    if not attentions:
        return None
    per_layer = []
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        per_layer.append(layer_attention.mean(dim=1).mean(dim=1))
    if not per_layer:
        return None
    stacked = torch.stack(per_layer, dim=0)
    return stacked.mean(dim=0)


def build_recency_indices(seq_len: int, keep_last_tokens: int) -> list[int]:
    if keep_last_tokens >= seq_len:
        return list(range(seq_len))
    start = max(seq_len - keep_last_tokens, 0)
    return list(range(start, seq_len))


def build_attention_indices(
    attention_scores: torch.Tensor,
    *,
    keep_last_tokens: int,
    keep_top_tokens: int,
    special_token_mask: torch.Tensor | None = None,
) -> list[int]:
    if attention_scores.dim() != 2:
        raise ValueError("attention_scores must have shape [batch, seq_len]")
    if attention_scores.shape[0] != 1:
        raise ValueError("attention-aware selection currently expects batch size 1")

    seq_len = attention_scores.shape[-1]
    recency = set(build_recency_indices(seq_len, keep_last_tokens))
    eligible = torch.arange(seq_len, device=attention_scores.device)
    scores = attention_scores[0].clone()

    if recency:
        recency_tensor = torch.tensor(sorted(recency), dtype=torch.long, device=scores.device)
        scores.index_fill_(0, recency_tensor, float("-inf"))

    if special_token_mask is not None:
        if special_token_mask.shape != attention_scores.shape:
            raise ValueError("special_token_mask must match attention_scores shape")
        special_positions = torch.nonzero(special_token_mask[0], as_tuple=False).flatten().tolist()
    else:
        special_positions = []

    k = min(keep_top_tokens, max(seq_len - len(recency), 0))
    top_indices = []
    if k > 0:
        top_indices = torch.topk(scores, k=k).indices.tolist()

    return sorted(recency.union(top_indices).union(special_positions))


@dataclass(slots=True)
class RecencyWindowPolicy(CompressionPolicy):
    keep_last_tokens: int
    name: str = "recency_window"

    def compress(self, past_key_values, context: PolicyContext | None = None) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)
        indices = build_recency_indices(original_tokens, self.keep_last_tokens)
        compressed = select_token_indices(past_key_values, indices)
        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed,
            original_tokens=original_tokens,
            compressed_tokens=len(indices),
            kept_token_indices=indices,
            metadata={"keep_last_tokens": self.keep_last_tokens},
        )


@dataclass(slots=True)
class AttentionRetentionPolicy(CompressionPolicy):
    keep_last_tokens: int
    keep_top_tokens: int
    name: str = "attention_retention"

    def compress(self, past_key_values, context: PolicyContext | None = None) -> CompressionOutcome:
        if context is None or context.attention_scores is None:
            raise ValueError("AttentionRetentionPolicy requires attention_scores in PolicyContext")
        original_tokens = KVCacheInspector.sequence_length(past_key_values)
        indices = build_attention_indices(
            context.attention_scores,
            keep_last_tokens=self.keep_last_tokens,
            keep_top_tokens=self.keep_top_tokens,
            special_token_mask=context.special_token_mask,
        )
        compressed = select_token_indices(past_key_values, indices)
        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed,
            original_tokens=original_tokens,
            compressed_tokens=len(indices),
            kept_token_indices=indices,
            metadata={
                "keep_last_tokens": self.keep_last_tokens,
                "keep_top_tokens": self.keep_top_tokens,
            },
        )
