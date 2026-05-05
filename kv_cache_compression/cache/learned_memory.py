"""
Phase 5 — Learned Memory Tokens for KV-Cache Compression.

Overview
--------
All previous phases (1–4) are *heuristic*: they select or cluster existing
tokens based on recency, attention scores, or k-means geometry.

Phase 5 introduces **learned compression**: instead of selecting tokens,
we learn a small set of *virtual memory tokens* whose key/value vectors
are optimised to summarise the full KV cache as accurately as possible.

There are two sub-strategies:

1. **LearnedMemoryPolicy**  (gradient-free, inference-time only)
   Fits a small set of `n_memory` synthetic KV vectors by minimising the
   mean-squared reconstruction error between the attention output computed
   with the compressed cache and the original cache — using a fast iterative
   optimisation (Adam, ~50 steps) at inference time.
   No training required; works with any frozen causal LM.

2. **LearnedMemoryFineTunePolicy**  (fine-tune assisted)
   If you have a lightweight adapter (LoRA-style linear layer) trained to
   predict good memory token initialisations from the mean KV statistics,
   this policy uses that adapter as a warm-start and then refines with the
   same iterative optimisation above.  This dramatically reduces the number
   of optimisation steps needed at inference time.

Key properties
--------------
- Memory tokens are not from the original sequence — they are *synthetic*,
  learned to be maximally informative summaries.
- The number of memory tokens `n_memory` is the only compression knob;
  it directly controls the cache budget.
- Works with any HuggingFace causal LM without modifying the model.
- Degrades gracefully: if optimisation is skipped (steps=0) it falls back
  to the K-Means centroid initialisation used in HybridClusterPolicy.

References
----------
- Mu et al. (2023). "Learning to Compress Prompts with Gist Tokens."
  arXiv:2304.08467
- Chevalier et al. (2023). "Adapting Language Models to Compress Contexts."
  arXiv:2305.14788
- Wingate et al. (2022). "Prompt Compression and Contrastive Conditioning
  for Controllability and Toxicity Reduction." arXiv:2210.03162
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .kv_cache import (
    CompressionOutcome,
    KVCacheInspector,
    KVCacheTensor,
    select_token_indices,
)
from .policies import CompressionPolicy, PolicyContext


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _kmeans_centroids(
    features: torch.Tensor,      # [seq_len, d]
    n_clusters: int,
    iterations: int = 20,
) -> torch.Tensor:
    """
    Quick K-Means++ to produce `n_clusters` centroid vectors.
    Used to initialise learned memory tokens with meaningful starting points.

    Returns tensor of shape [n_clusters, d].
    """
    n, d = features.shape
    device = features.device

    if n <= n_clusters:
        # Pad with zeros if fewer tokens than clusters requested
        pad = torch.zeros(n_clusters - n, d, device=device, dtype=features.dtype)
        return torch.cat([features, pad], dim=0)

    # K-Means++ init
    first = torch.randint(0, n, (1,), device=device).item()
    centers = [features[first]]
    for _ in range(1, n_clusters):
        stack = torch.stack(centers, dim=0)              # [k, d]
        dists = torch.cdist(features, stack).min(dim=-1).values.pow(2)
        total = dists.sum()
        probs = dists / total if total > 0 else torch.ones(n, device=device) / n
        idx = int(torch.multinomial(probs, 1).item())
        centers.append(features[idx])
    centers_t = torch.stack(centers, dim=0)              # [n_clusters, d]

    # Lloyd iterations
    for _ in range(iterations):
        dists = torch.cdist(features, centers_t)         # [n, k]
        assign = dists.argmin(dim=-1)                    # [n]
        new_centers = []
        for ci in range(n_clusters):
            members = features[assign == ci]
            new_centers.append(members.mean(dim=0) if members.numel() > 0 else centers_t[ci])
        centers_t = torch.stack(new_centers, dim=0)

    return centers_t                                     # [n_clusters, d]


def _kv_to_features(past_key_values: KVCacheTensor) -> torch.Tensor:
    """
    Flatten the last layer's key+value tensors into a 2-D feature matrix.

    Shape: [seq_len, 2 * head_dim] (averaged over heads, batch=1 assumed).
    """
    last_key, last_val = past_key_values[-1]    # [1, heads, seq, head_dim]
    key_feat = last_key[0].mean(dim=0)          # [seq, head_dim]
    val_feat = last_val[0].mean(dim=0)          # [seq, head_dim]
    return torch.cat([key_feat, val_feat], dim=-1)  # [seq, 2*head_dim]


def _attention_output_approx(
    query: torch.Tensor,      # [1, heads, 1, head_dim]
    key: torch.Tensor,        # [1, heads, seq, head_dim]
    value: torch.Tensor,      # [1, heads, seq, head_dim]
    scale: float | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention output (no masking — memory tokens are global).
    Returns [1, heads, 1, head_dim].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(key.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale   # [1, h, 1, seq]
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)                            # [1, h, 1, hd]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LearnedMemoryPolicy  (inference-time optimisation, no training needed)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedMemoryPolicy(CompressionPolicy):
    """
    Compress the KV cache into `n_memory` learned synthetic memory tokens.

    At inference time, for each layer we:
      1. Initialise `n_memory` synthetic key/value vectors via K-Means++
         on the existing KV tensors (warm start for fast convergence).
      2. Optimise those vectors for `optim_steps` gradient steps to minimise
         the mean-squared error between:
             attention_output(original_cache)  vs
             attention_output(memory_tokens)
         using a representative query (the mean query of the last layer).
      3. Replace the full KV cache with just the `n_memory` optimised vectors.

    This approach is inspired by *gist token* and *context compression* work
    (Mu et al. 2023, Chevalier et al. 2023) but operates purely at inference
    time — no fine-tuning is required.

    Parameters
    ----------
    n_memory      : number of synthetic memory tokens (the compression budget)
    optim_steps   : gradient steps to optimise memory tokens per call
    lr            : Adam learning rate for memory token optimisation
    use_kmeans_init: if True, warm-start memory tokens with K-Means++ centroids
                    (strongly recommended; reduces optim_steps needed)
    always_keep_last : always preserve the last N real tokens unchanged
                       (recency anchors — helps with local coherence)
    name          : policy name for reporting
    """
    n_memory: int = 32
    optim_steps: int = 50
    lr: float = 1e-2
    use_kmeans_init: bool = True
    always_keep_last: int = 64
    name: str = "learned_memory"

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)

        # ── Edge case: nothing to compress ──
        if original_tokens <= self.n_memory + self.always_keep_last:
            return CompressionOutcome(
                policy_name=self.name,
                past_key_values=past_key_values,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                kept_token_indices=list(range(original_tokens)),
                metadata={"skipped": True, "reason": "cache already fits budget"},
            )

        num_layers = KVCacheInspector.num_layers(past_key_values)
        device = past_key_values[0][0].device
        orig_dtype = past_key_values[0][0].dtype

        # ── Split: tokens to compress vs recency anchor ──
        compress_end = original_tokens - self.always_keep_last
        # Recency: always-kept real tokens at the end
        recency_indices = list(range(compress_end, original_tokens))

        # ── Initialise memory tokens per layer ──
        if self.use_kmeans_init:
            # Use KV features from the last layer for fast centroid init
            features = _kv_to_features(past_key_values)[:compress_end]  # [compress_end, d]
            centroids = _kmeans_centroids(
                features.float(),
                n_clusters=self.n_memory,
            )   # [n_memory, 2*head_dim]
            half_d = centroids.shape[-1] // 2
            key_init_global = centroids[:, :half_d]   # [n_memory, head_dim]
            val_init_global = centroids[:, half_d:]   # [n_memory, head_dim]
        else:
            key_init_global = None
            val_init_global = None

        # ── Optimise memory tokens layer by layer ──
        memory_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, (key, value) in enumerate(past_key_values):
            # key / value shape: [1, heads, seq, head_dim]
            _, num_heads, _, head_dim = key.shape

            # Slice the tokens to compress
            key_src = key[..., :compress_end, :].detach()   # [1, h, compress_end, hd]
            val_src = value[..., :compress_end, :].detach()

            # ── Initialise synthetic memory KV tensors ──
            if key_init_global is not None:
                # Broadcast centroid init across all heads
                # centroids were computed in averaged-head space → expand to per-head
                k_init = key_init_global.unsqueeze(0).unsqueeze(0).expand(
                    1, num_heads, self.n_memory, head_dim
                ).to(device=device, dtype=torch.float32).clone()
                v_init = val_init_global.unsqueeze(0).unsqueeze(0).expand(
                    1, num_heads, self.n_memory, head_dim
                ).to(device=device, dtype=torch.float32).clone()
            else:
                k_init = torch.randn(1, num_heads, self.n_memory, head_dim,
                                     device=device, dtype=torch.float32) * 0.02
                v_init = torch.randn(1, num_heads, self.n_memory, head_dim,
                                     device=device, dtype=torch.float32) * 0.02

            mem_key = nn.Parameter(k_init)
            mem_val = nn.Parameter(v_init)
            optimizer = torch.optim.Adam([mem_key, mem_val], lr=self.lr)

            # Representative query: mean over the sequence & heads of the original key
            # (a lightweight proxy — we don't have the real query vectors here)
            query_repr = key_src.float().mean(dim=-2, keepdim=True)  # [1, h, 1, hd]
            scale = 1.0 / math.sqrt(head_dim)

            # ── Optimisation loop ──
            if self.optim_steps > 0:
                key_src_f = key_src.float()
                val_src_f = val_src.float()

                # Target: attention output using the *original* compressed tokens
                with torch.no_grad():
                    target_out = _attention_output_approx(
                        query_repr, key_src_f, val_src_f, scale=scale
                    )   # [1, h, 1, hd]

                for _ in range(self.optim_steps):
                    optimizer.zero_grad()
                    approx_out = _attention_output_approx(
                        query_repr, mem_key, mem_val, scale=scale
                    )
                    loss = torch.nn.functional.mse_loss(approx_out, target_out)
                    loss.backward()
                    optimizer.step()

            # ── Detach and cast back to original dtype ──
            opt_key = mem_key.detach().to(dtype=orig_dtype)   # [1, h, n_memory, hd]
            opt_val = mem_val.detach().to(dtype=orig_dtype)

            # ── Concat with recency tokens ──
            recency_key = key[..., compress_end:, :]          # [1, h, always_keep_last, hd]
            recency_val = value[..., compress_end:, :]

            final_key = torch.cat([opt_key, recency_key], dim=-2)   # [1,h,n_mem+last,hd]
            final_val = torch.cat([opt_val, recency_val], dim=-2)

            memory_layers.append((final_key, final_val))

        compressed_cache = tuple(memory_layers)
        compressed_tokens = self.n_memory + len(recency_indices)

        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed_cache,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            # Memory tokens have synthetic positions; use negative indices as placeholder
            kept_token_indices=list(range(-self.n_memory, 0)) + recency_indices,
            metadata={
                "n_memory": self.n_memory,
                "optim_steps": self.optim_steps,
                "lr": self.lr,
                "use_kmeans_init": self.use_kmeans_init,
                "always_keep_last": self.always_keep_last,
                "tokens_compressed_into_memory": compress_end,
                "num_layers_optimised": num_layers,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MemoryTokenAdapter  — lightweight trained warm-start predictor
# ══════════════════════════════════════════════════════════════════════════════

class MemoryTokenAdapter(nn.Module):
    """
    A tiny MLP that predicts good initial memory token KV vectors from
    a compact statistical summary of the full KV cache.

    Architecture:
        input  : [mean_key | mean_val | std_key | std_val]   shape [4 * head_dim]
        hidden : ReLU MLP with `hidden_dim` units
        output : [n_memory * 2 * head_dim]  (flattened key+value for all memory tokens)

    Training signal: reconstruction MSE between predicted memory KV vectors
    (after a single attention output pass) and the ground-truth attention output
    computed with the full cache.

    This module is designed to be trained offline on a small representative
    corpus (a few thousand prompts suffice) and then used as a plug-in warm-start
    inside `LearnedMemoryFineTunePolicy`.
    """

    def __init__(
        self,
        head_dim: int,
        n_memory: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_memory = n_memory

        input_dim  = 4 * head_dim          # mean_k, mean_v, std_k, std_v
        output_dim = n_memory * 2 * head_dim  # flattened [n_mem, key+val]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict initial memory token KVs from full-cache statistics.

        Parameters
        ----------
        key   : [1, heads, seq, head_dim]
        value : [1, heads, seq, head_dim]

        Returns
        -------
        mem_key : [1, heads, n_memory, head_dim]
        mem_val : [1, heads, n_memory, head_dim]
        """
        # Aggregate over heads and sequence → compact summary
        k_flat = key[0]    # [heads, seq, hd]
        v_flat = value[0]  # [heads, seq, hd]

        mean_k = k_flat.mean(dim=(0, 1))   # [hd]
        mean_v = v_flat.mean(dim=(0, 1))   # [hd]
        std_k  = k_flat.std(dim=(0, 1))    # [hd]
        std_v  = v_flat.std(dim=(0, 1))    # [hd]

        stats = torch.cat([mean_k, mean_v, std_k, std_v], dim=-1).unsqueeze(0)  # [1, 4*hd]
        out = self.mlp(stats.float())           # [1, n_mem * 2 * hd]

        n_heads = key.shape[1]
        out = out.view(self.n_memory, 2, self.head_dim)   # [n_mem, 2, hd]
        mem_key = out[:, 0, :].unsqueeze(0).unsqueeze(0).expand(1, n_heads, -1, -1)
        mem_val = out[:, 1, :].unsqueeze(0).unsqueeze(0).expand(1, n_heads, -1, -1)
        return mem_key.contiguous(), mem_val.contiguous()

    def compute_loss(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loss: MSE between attention output using predicted memory tokens
        vs attention output using the full original cache.

        Use this in a training loop:
            optim.zero_grad()
            loss = adapter.compute_loss(key, value)
            loss.backward()
            optim.step()
        """
        import math
        scale = 1.0 / math.sqrt(self.head_dim)

        mem_key, mem_val = self.forward(key, value)
        query = key.float().mean(dim=-2, keepdim=True)        # [1, h, 1, hd]

        with torch.no_grad():
            target = _attention_output_approx(
                query, key.float(), value.float(), scale=scale
            )
        pred = _attention_output_approx(
            query, mem_key.float(), mem_val.float(), scale=scale
        )
        return torch.nn.functional.mse_loss(pred, target)

    def save(self, path: str) -> None:
        """Save adapter weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"[MemoryTokenAdapter] Saved to {path}")

    @classmethod
    def load(cls, path: str, head_dim: int, n_memory: int, **kwargs) -> "MemoryTokenAdapter":
        """Load adapter weights from disk."""
        adapter = cls(head_dim=head_dim, n_memory=n_memory, **kwargs)
        adapter.load_state_dict(torch.load(path, map_location="cpu"))
        adapter.eval()
        print(f"[MemoryTokenAdapter] Loaded from {path}")
        return adapter


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LearnedMemoryFineTunePolicy  (adapter-assisted warm start)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedMemoryFineTunePolicy(CompressionPolicy):
    """
    Adapter-assisted learned memory compression.

    Uses a pre-trained `MemoryTokenAdapter` to predict good initial KV vectors
    for the memory tokens, then refines them with a small number of gradient
    steps (typically 5–10 instead of 50 needed without the adapter).

    Workflow:
        1. Adapter forward pass  →  warm-start memory KV tensors       (fast)
        2. Adam optimisation for `optim_steps` steps                  (few steps)
        3. Concat with recency tokens                                  (same as base)

    Training the adapter (offline, one-time):
        See `MemoryTokenAdapter.compute_loss` and the training script at
        `experiments/train_memory_adapter.py`.

    Parameters
    ----------
    adapter       : pre-trained MemoryTokenAdapter instance
    n_memory      : number of synthetic memory tokens
    optim_steps   : refinement steps after adapter warm-start (typically 5–15)
    lr            : Adam learning rate for refinement
    always_keep_last : always preserve the last N real tokens unchanged
    name          : policy name for reporting
    """
    adapter: "MemoryTokenAdapter"
    n_memory: int = 32
    optim_steps: int = 10
    lr: float = 5e-3
    always_keep_last: int = 64
    name: str = "learned_memory_ft"

    def compress(
        self,
        past_key_values: KVCacheTensor,
        context: PolicyContext | None = None,
    ) -> CompressionOutcome:
        original_tokens = KVCacheInspector.sequence_length(past_key_values)

        if original_tokens <= self.n_memory + self.always_keep_last:
            return CompressionOutcome(
                policy_name=self.name,
                past_key_values=past_key_values,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                kept_token_indices=list(range(original_tokens)),
                metadata={"skipped": True, "reason": "cache already fits budget"},
            )

        num_layers = KVCacheInspector.num_layers(past_key_values)
        device = past_key_values[0][0].device
        orig_dtype = past_key_values[0][0].dtype
        compress_end = original_tokens - self.always_keep_last
        recency_indices = list(range(compress_end, original_tokens))

        self.adapter.to(device).eval()

        memory_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, (key, value) in enumerate(past_key_values):
            _, num_heads, _, head_dim = key.shape
            key_src = key[..., :compress_end, :].detach()
            val_src = value[..., :compress_end, :].detach()

            # ── Adapter warm-start ──
            with torch.no_grad():
                mem_key_init, mem_val_init = self.adapter(key_src, val_src)
            # mem_key_init: [1, heads, n_memory, head_dim]

            mem_key = nn.Parameter(mem_key_init.float().to(device))
            mem_val = nn.Parameter(mem_val_init.float().to(device))
            optimizer = torch.optim.Adam([mem_key, mem_val], lr=self.lr)

            scale = 1.0 / math.sqrt(head_dim)
            query_repr = key_src.float().mean(dim=-2, keepdim=True)

            if self.optim_steps > 0:
                with torch.no_grad():
                    target_out = _attention_output_approx(
                        query_repr, key_src.float(), val_src.float(), scale=scale
                    )
                for _ in range(self.optim_steps):
                    optimizer.zero_grad()
                    approx_out = _attention_output_approx(
                        query_repr, mem_key, mem_val, scale=scale
                    )
                    loss = torch.nn.functional.mse_loss(approx_out, target_out)
                    loss.backward()
                    optimizer.step()

            opt_key = mem_key.detach().to(dtype=orig_dtype)
            opt_val = mem_val.detach().to(dtype=orig_dtype)

            recency_key = key[..., compress_end:, :]
            recency_val = value[..., compress_end:, :]
            final_key = torch.cat([opt_key, recency_key], dim=-2)
            final_val = torch.cat([opt_val, recency_val], dim=-2)
            memory_layers.append((final_key, final_val))

        compressed_cache = tuple(memory_layers)
        compressed_tokens = self.n_memory + len(recency_indices)

        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed_cache,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            kept_token_indices=list(range(-self.n_memory, 0)) + recency_indices,
            metadata={
                "n_memory": self.n_memory,
                "optim_steps": self.optim_steps,
                "lr": self.lr,
                "always_keep_last": self.always_keep_last,
                "adapter_used": True,
                "tokens_compressed_into_memory": compress_end,
                "num_layers_optimised": num_layers,
            },
        )
