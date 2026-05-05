from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_cache import CompressionOutcome, KVCacheInspector, KVCacheTensor, merge_token_segments, select_token_indices
from .policies import CompressionPolicy, PolicyContext
from .prune import build_attention_indices


def _token_features_from_cache(past_key_values: KVCacheTensor) -> torch.Tensor:
    if not past_key_values:
        raise ValueError("past_key_values must not be empty")
    last_key, last_value = past_key_values[-1]
    if last_key.shape[0] != 1:
        raise ValueError("Hybrid clustering currently expects batch size 1")
    key_features = last_key[0].mean(dim=0)
    value_features = last_value[0].mean(dim=0)
    return torch.cat([key_features, value_features], dim=-1)


def _kmeans_plus_plus_init(features: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    K-means++ initialisation (Arthur & Vassilvitskii, 2007).

    Selects initial centroids with probability proportional to squared distance
    from the nearest already-chosen centroid. This dramatically reduces the
    number of iterations needed vs. uniform random / linspace initialisation
    and consistently yields lower inertia (tighter, more meaningful clusters).
    """
    n = features.shape[0]
    device = features.device

    # Pick first centroid uniformly at random
    first = torch.randint(0, n, (1,), device=device).item()
    centers = [features[first]]

    for _ in range(1, num_clusters):
        center_stack = torch.stack(centers, dim=0)          # [k, d]
        # Squared distance from each point to nearest center
        dists = torch.cdist(features, center_stack)         # [n, k]
        min_dists_sq = dists.min(dim=-1).values.pow(2)      # [n]

        # Sample proportional to squared distance
        total = min_dists_sq.sum()
        if total == 0:
            # All points coincide — pick random
            idx = torch.randint(0, n, (1,), device=device).item()
        else:
            probs = min_dists_sq / total
            idx = int(torch.multinomial(probs, num_samples=1).item())
        centers.append(features[idx])

    return torch.stack(centers, dim=0)   # [num_clusters, d]


def _simple_kmeans(
    features: torch.Tensor,
    num_clusters: int,
    iterations: int = 30,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    K-means with k-means++ initialisation and early-stopping via centroid drift.

    Improvements over previous implementation:
      - k-means++ seeding → better initial centroids, fewer iterations needed
      - Convergence tolerance: stops early when centroid movement < `tol`
      - Max iterations increased to 30 (converges faster due to better init)
    """
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if features.shape[0] <= num_clusters:
        return torch.arange(features.shape[0], device=features.device)

    centers = _kmeans_plus_plus_init(features, num_clusters)   # [k, d]

    assignments = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    for _ in range(iterations):
        distances = torch.cdist(features, centers)             # [n, k]
        new_assignments = distances.argmin(dim=-1)             # [n]

        updated_centers = []
        for cluster_idx in range(num_clusters):
            members = features[new_assignments == cluster_idx]
            if members.numel() == 0:
                updated_centers.append(centers[cluster_idx])
            else:
                updated_centers.append(members.mean(dim=0))
        new_centers = torch.stack(updated_centers, dim=0)

        # Early stopping: check centroid drift
        drift = (new_centers - centers).norm(dim=-1).max().item()
        assignments = new_assignments
        centers = new_centers
        if drift < tol:
            break

    return assignments


def _average_segment(past_key_values: KVCacheTensor, token_indices: list[int]) -> KVCacheTensor:
    segment = select_token_indices(past_key_values, token_indices)
    averaged_layers = []
    for key, value in segment:
        averaged_layers.append((key.mean(dim=-2, keepdim=True), value.mean(dim=-2, keepdim=True)))
    return tuple(averaged_layers)


@dataclass(slots=True)
class HybridClusterPolicy(CompressionPolicy):
    keep_last_tokens: int
    keep_top_tokens: int
    cluster_tokens: int
    name: str = "hybrid_cluster"

    def compress(self, past_key_values, context: PolicyContext | None = None) -> CompressionOutcome:
        if context is None or context.attention_scores is None:
            raise ValueError("HybridClusterPolicy requires attention_scores in PolicyContext")

        original_tokens = KVCacheInspector.sequence_length(past_key_values)
        exact_keep = set(
            build_attention_indices(
                context.attention_scores,
                keep_last_tokens=self.keep_last_tokens,
                keep_top_tokens=self.keep_top_tokens,
                special_token_mask=context.special_token_mask,
            )
        )
        stale_indices = [idx for idx in range(original_tokens) if idx not in exact_keep]
        if not stale_indices or self.cluster_tokens <= 0:
            compressed = select_token_indices(past_key_values, sorted(exact_keep))
            kept = sorted(exact_keep)
            return CompressionOutcome(
                policy_name=self.name,
                past_key_values=compressed,
                original_tokens=original_tokens,
                compressed_tokens=len(kept),
                kept_token_indices=kept,
                metadata={"clusters": 0, "cluster_tokens": self.cluster_tokens},
            )

        features = _token_features_from_cache(past_key_values)[stale_indices]
        num_clusters = min(self.cluster_tokens, len(stale_indices))
        assignments = _simple_kmeans(features, num_clusters=num_clusters)

        segments: list[tuple[int, KVCacheTensor]] = []
        for idx in sorted(exact_keep):
            segments.append((idx, _average_segment(past_key_values, [idx])))

        cluster_count = 0
        for cluster_idx in range(num_clusters):
            member_offsets = torch.nonzero(assignments == cluster_idx, as_tuple=False).flatten().tolist()
            if not member_offsets:
                continue
            member_indices = [stale_indices[offset] for offset in member_offsets]
            representative_position = min(member_indices)
            segments.append((representative_position, _average_segment(past_key_values, member_indices)))
            cluster_count += 1

        compressed_cache, kept_positions = merge_token_segments(segments)
        return CompressionOutcome(
            policy_name=self.name,
            past_key_values=compressed_cache,
            original_tokens=original_tokens,
            compressed_tokens=KVCacheInspector.sequence_length(compressed_cache),
            kept_token_indices=kept_positions,
            metadata={
                "keep_last_tokens": self.keep_last_tokens,
                "keep_top_tokens": self.keep_top_tokens,
                "cluster_tokens": self.cluster_tokens,
                "realized_clusters": cluster_count,
            },
        )
