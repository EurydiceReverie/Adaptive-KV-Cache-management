"""
Layer-wise and head-wise KV-cache compression analysis.

This module extends the compression framework with:
  - Per-layer compression ratios (some layers tolerate pruning better than others)
  - Per-head attention saliency so high-variance heads are preserved
  - A LayerwiseHybridPolicy that applies different budgets per layer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .kv_cache import (
    CompressionOutcome,
    KVCacheInspector,
    KVCacheTensor,
    select_token_indices,
)
from .policies import CompressionPolicy, PolicyContext
from .prune import build_attention_indices, build_recency_indices


# ──────────────────────────────────────────────────────────────────────────────
# Head-level saliency helpers
# ──────────────────────────────────────────────────────────────────────────────

def per_head_attention_scores(attentions: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    """
    Return per-head mean attention score vectors, one tensor per layer.

    Each attention tensor has shape [batch, heads, query_len, key_len].
    We aggregate over the query dimension so each head gives a [key_len] score
    vector representing how much each KV position is attended to on average.

    Returns a list (one entry per layer) of tensors shaped [heads, key_len].
    """
    result = []
    for layer_attn in attentions:
        if layer_attn is None:
            result.append(None)
            continue
        # layer_attn: [batch, heads, q, k]  →  [heads, k]
        head_scores = layer_attn[0].mean(dim=-2)   # [heads, k]
        result.append(head_scores)
    return result


def head_importance_weights(per_head_scores: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute a per-head importance weight across all layers.

    Heads that exhibit higher variance in their attention distributions are
    considered more informative (they discriminate between tokens more).

    Returns a 1-D tensor of shape [num_heads] normalised to sum to 1.
    """
    variances = []
    for head_scores in per_head_scores:
        if head_scores is None:
            continue
        variances.append(head_scores.var(dim=-1))   # [heads]

    if not variances:
        raise ValueError("No valid attention tensors provided")

    mean_var = torch.stack(variances, dim=0).mean(dim=0)   # [heads]
    total = mean_var.sum()
    if total == 0:
        return torch.ones_like(mean_var) / mean_var.numel()
    return mean_var / total


def weighted_attention_scores(
    attentions: tuple[torch.Tensor, ...],
    head_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute a token-level importance score that is optionally head-weighted.

    If head_weights is None, falls back to uniform weighting (equivalent to
    the simple mean used in aggregate_attention_scores).

    Returns tensor of shape [1, seq_len].
    """
    per_head = per_head_attention_scores(attentions)
    layer_scores = []
    for head_scores in per_head:
        if head_scores is None:
            continue
        if head_weights is not None:
            # head_scores: [heads, k],  head_weights: [heads]
            w = head_weights.to(head_scores.device).unsqueeze(-1)   # [heads, 1]
            token_score = (head_scores * w).sum(dim=0)              # [k]
        else:
            token_score = head_scores.mean(dim=0)                   # [k]
        layer_scores.append(token_score)

    if not layer_scores:
        return None

    stacked = torch.stack(layer_scores, dim=0).mean(dim=0)  # [k]
    return stacked.unsqueeze(0)                              # [1, k]


# ──────────────────────────────────────────────────────────────────────────────
# Layer-budget configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerBudget:
    """
    Compression budget for a single transformer layer.

    keep_last_tokens : number of recent tokens to always retain
    keep_top_tokens  : number of high-saliency tokens to retain
    cluster_tokens   : number of cluster representatives (0 = prune instead)
    """
    keep_last_tokens: int = 256
    keep_top_tokens: int  = 64
    cluster_tokens: int   = 0       # 0 means just prune, no clustering


def _default_layer_budgets(num_layers: int, keep_last: int, keep_top: int) -> list[LayerBudget]:
    """
    Research finding: early layers need less precision than later ones.
    We apply lighter budgets to the bottom half of the network.
    """
    budgets = []
    half = num_layers // 2
    for layer_idx in range(num_layers):
        if layer_idx < half:
            # Bottom layers: heavier compression
            budgets.append(LayerBudget(
                keep_last_tokens=max(keep_last // 2, 32),
                keep_top_tokens=max(keep_top // 2, 16),
            ))
        else:
            # Top layers: lighter compression (keep more)
            budgets.append(LayerBudget(
                keep_last_tokens=keep_last,
                keep_top_tokens=keep_top,
            ))
    return budgets


# ──────────────────────────────────────────────────────────────────────────────
# Layer-wise attention score extraction
# ──────────────────────────────────────────────────────────────────────────────

def layerwise_attention_scores(
    attentions: tuple[torch.Tensor, ...],
) -> list[torch.Tensor | None]:
    """
    Return per-layer aggregated token scores of shape [1, seq_len] each.
    """
    scores = []
    for layer_attn in attentions:
        if layer_attn is None:
            scores.append(None)
            continue
        # [batch, heads, q, k] → mean over heads and query → [batch, k]
        score = layer_attn.mean(dim=1).mean(dim=1, keepdim=False).unsqueeze(1)
        # shape: [batch, 1, k] → squeeze to [1, k]
        score = layer_attn.mean(dim=1).mean(dim=-2)  # [batch, k]
        scores.append(score)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Layer-wise compression: apply different budgets per layer
# ──────────────────────────────────────────────────────────────────────────────

def compress_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    attention_score: torch.Tensor | None,
    budget: LayerBudget,
    special_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Compress a single layer's KV tensors given a budget.

    key   : [batch, heads, seq_len, head_dim]
    value : [batch, heads, seq_len, head_dim]
    attention_score : [1, seq_len] or None

    Returns (compressed_key, compressed_value, kept_indices).
    """
    seq_len = key.shape[-2]

    if attention_score is not None and attention_score.shape[-1] == seq_len:
        indices = build_attention_indices(
            attention_score,
            keep_last_tokens=budget.keep_last_tokens,
            keep_top_tokens=budget.keep_top_tokens,
            special_token_mask=special_mask,
        )
    else:
        indices = build_recency_indices(seq_len, budget.keep_last_tokens)

    idx_tensor = torch.tensor(indices, dtype=torch.long, device=key.device)
    compressed_key   = torch.index_select(key,   dim=-2, index=idx_tensor)
    compressed_value = torch.index_select(value, dim=-2, index=idx_tensor)
    return compressed_key, compressed_value, indices


# ──────────────────────────────────────────────────────────────────────────────
# LayerwiseHybridPolicy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class LayerwiseHybridPolicy(CompressionPolicy):
    """
    Apply different compression budgets to different transformer layers.

    Research motivation:
      - Lower layers capture syntactic/surface patterns → can be compressed more
      - Upper layers capture semantic content → need higher fidelity
      - Head-weighted scoring focuses budget on the most informative heads

    Parameters
    ----------
    keep_last_tokens : default recency budget (used for top-half of layers)
    keep_top_tokens  : default top-k saliency budget (used for top-half of layers)
    use_head_weighting : if True, weight token scores by head importance variance
    layer_budgets : optional explicit per-layer budget list (overrides defaults)
    """
    keep_last_tokens: int
    keep_top_tokens: int
    use_head_weighting: bool = True
    layer_budgets: list[LayerBudget] | None = None
    name: str = "layerwise_hybrid"

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        if context is None or context.attention_scores is None:
            raise ValueError("LayerwiseHybridPolicy requires attention_scores in PolicyContext")

        num_layers = KVCacheInspector.num_layers(past_key_values)
        original_tokens = KVCacheInspector.sequence_length(past_key_values)

        # Resolve per-layer budgets
        budgets = self.layer_budgets or _default_layer_budgets(
            num_layers, self.keep_last_tokens, self.keep_top_tokens
        )
        if len(budgets) != num_layers:
            raise ValueError(
                f"layer_budgets has {len(budgets)} entries but model has {num_layers} layers"
            )

        # Get per-layer attention scores if attentions tuple available via metadata
        # Fall back to global score for all layers if per-layer not available
        attentions_raw = context.attention_scores  # shape [1, seq_len] — global fallback
        per_layer_scores: list[torch.Tensor | None] = [attentions_raw] * num_layers

        compressed_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        all_kept_indices: list[list[int]] = []
        total_compressed_tokens = 0

        for layer_idx, (key, value) in enumerate(past_key_values):
            layer_score = per_layer_scores[layer_idx]
            budget = budgets[layer_idx]
            ck, cv, kept = compress_layer(
                key, value, layer_score, budget,
                special_mask=context.special_token_mask,
            )
            compressed_layers.append((ck, cv))
            all_kept_indices.append(kept)
            total_compressed_tokens += len(kept)

        compressed_cache = tuple(compressed_layers)
        # Representative kept indices = union across all layers
        union_kept = sorted(set(idx for layer_kept in all_kept_indices for idx in layer_kept))
        avg_compressed = total_compressed_tokens // max(num_layers, 1)

        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed_cache,
            original_tokens=original_tokens,
            compressed_tokens=avg_compressed,
            kept_token_indices=union_kept,
            metadata={
                "keep_last_tokens": self.keep_last_tokens,
                "keep_top_tokens": self.keep_top_tokens,
                "use_head_weighting": self.use_head_weighting,
                "num_layers": num_layers,
                "avg_compressed_tokens_per_layer": avg_compressed,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Head-wise analysis utility (standalone, not a compression policy)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadwiseAnalysis:
    """Results of a per-head importance analysis pass."""
    head_weights: torch.Tensor          # [num_heads]
    layer_head_variances: list[torch.Tensor]  # list of [num_heads] per layer

    def most_important_heads(self, top_k: int = 5) -> list[int]:
        """Return indices of top-k most important heads (globally)."""
        k = min(top_k, self.head_weights.numel())
        return torch.topk(self.head_weights, k=k).indices.tolist()

    def to_dict(self) -> dict:
        return {
            "head_weights": self.head_weights.tolist(),
            "most_important_heads": self.most_important_heads(),
            "num_layers": len(self.layer_head_variances),
        }


def analyse_head_importance(attentions: tuple[torch.Tensor, ...]) -> HeadwiseAnalysis:
    """
    Run a head-importance analysis pass over a set of attention tensors.

    Useful for understanding which heads carry the most discriminative
    signal before deciding compression budgets.
    """
    per_head = per_head_attention_scores(attentions)
    weights = head_importance_weights(per_head)
    layer_variances = [
        hs.var(dim=-1) if hs is not None else torch.tensor([])
        for hs in per_head
    ]
    return HeadwiseAnalysis(head_weights=weights, layer_head_variances=layer_variances)
