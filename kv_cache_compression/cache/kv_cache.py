from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

KVCacheTensor = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def estimate_tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def to_legacy_tuple(past_key_values) -> KVCacheTensor:
    """
    Convert any KV cache format to a plain tuple[tuple[Tensor, Tensor], ...].

    Handles all known transformers versions:
      - transformers < 4.38 : plain tuple of (key, value) pairs
      - transformers 4.38–5.x : DynamicCache with .key_cache / .value_cache lists
      - transformers 5.x+    : DynamicCache with .layers list of DynamicLayer objects
                               each having .keys and .values tensors
    """
    # ── transformers 5.x: DynamicCache with .layers (list of DynamicLayer) ──
    if hasattr(past_key_values, "layers") and isinstance(
        getattr(past_key_values, "layers", None), list
    ):
        layers = past_key_values.layers
        if len(layers) == 0:
            return tuple()
        if hasattr(layers[0], "keys") and hasattr(layers[0], "values"):
            return tuple((layer.keys, layer.values) for layer in layers)

    # ── transformers 4.38–4.x: DynamicCache with .key_cache / .value_cache ──
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = past_key_values.key_cache
        val_cache = past_key_values.value_cache
        if len(key_cache) == 0:
            return tuple()
        return tuple((k, v) for k, v in zip(key_cache, val_cache))

    # ── Legacy: plain tuple/list of (key_tensor, value_tensor) pairs ──
    if isinstance(past_key_values, (tuple, list)):
        result = []
        for item in past_key_values:
            if isinstance(item, (tuple, list)):
                tensors = list(item)
                if len(tensors) == 2 and all(isinstance(t, torch.Tensor) for t in tensors):
                    result.append((tensors[0], tensors[1]))
                else:
                    raise TypeError(
                        f"Expected (key_tensor, value_tensor) pair, got {[type(t) for t in tensors]}"
                    )
            else:
                raise TypeError(
                    f"Unexpected item type {type(item)} in past_key_values list."
                )
        return tuple(result)

    raise TypeError(
        f"Cannot convert past_key_values of type {type(past_key_values)} to legacy tuple. "
        f"Attributes: {[a for a in dir(past_key_values) if not a.startswith('_')][:15]}"
    )


@dataclass(slots=True)
class CompressionOutcome:
    policy_name: str
    past_key_values: KVCacheTensor
    original_tokens: int
    compressed_tokens: int
    kept_token_indices: list[int]
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def token_reduction(self) -> int:
        return self.original_tokens - self.compressed_tokens


class KVCacheInspector:
    @staticmethod
    def sequence_length(past_key_values: KVCacheTensor) -> int:
        if not past_key_values:
            return 0
        return int(past_key_values[0][0].shape[-2])

    @staticmethod
    def num_layers(past_key_values: KVCacheTensor) -> int:
        return len(past_key_values)

    @staticmethod
    def total_bytes(past_key_values: KVCacheTensor) -> int:
        total = 0
        for key, value in past_key_values:
            total += estimate_tensor_bytes(key)
            total += estimate_tensor_bytes(value)
        return total

    @staticmethod
    def layerwise_bytes(past_key_values: KVCacheTensor) -> list[int]:
        return [estimate_tensor_bytes(key) + estimate_tensor_bytes(value) for key, value in past_key_values]

    @staticmethod
    def summary(past_key_values: KVCacheTensor) -> dict[str, int | list[int]]:
        return {
            "layers": KVCacheInspector.num_layers(past_key_values),
            "tokens": KVCacheInspector.sequence_length(past_key_values),
            "total_bytes": KVCacheInspector.total_bytes(past_key_values),
            "layerwise_bytes": KVCacheInspector.layerwise_bytes(past_key_values),
        }


def _normalize_indices(indices: Iterable[int], seq_len: int, device: torch.device) -> torch.Tensor:
    unique_sorted = sorted({idx for idx in indices if 0 <= idx < seq_len})
    return torch.tensor(unique_sorted, dtype=torch.long, device=device)


def select_token_indices(past_key_values: KVCacheTensor, indices: Sequence[int]) -> KVCacheTensor:
    if not past_key_values:
        return past_key_values
    seq_len = KVCacheInspector.sequence_length(past_key_values)
    index_tensor = _normalize_indices(indices, seq_len, past_key_values[0][0].device)
    selected: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in past_key_values:
        selected.append((
            torch.index_select(key, dim=-2, index=index_tensor),
            torch.index_select(value, dim=-2, index=index_tensor),
        ))
    return tuple(selected)


def merge_token_segments(segments: Sequence[tuple[int, KVCacheTensor]]) -> tuple[KVCacheTensor, list[int]]:
    if not segments:
        return tuple(), []
    ordered = sorted(segments, key=lambda item: item[0])
    kept_positions = [position for position, _ in ordered]
    layer_buckets: list[list[torch.Tensor]] = [[] for _ in range(len(ordered[0][1]))]
    value_buckets: list[list[torch.Tensor]] = [[] for _ in range(len(ordered[0][1]))]

    for _, segment in ordered:
        for layer_idx, (key, value) in enumerate(segment):
            layer_buckets[layer_idx].append(key)
            value_buckets[layer_idx].append(value)

    merged_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_keys, layer_values in zip(layer_buckets, value_buckets):
        merged_layers.append((torch.cat(layer_keys, dim=-2), torch.cat(layer_values, dim=-2)))
    return tuple(merged_layers), kept_positions
