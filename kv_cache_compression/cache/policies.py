from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .kv_cache import CompressionOutcome, KVCacheTensor


@dataclass(slots=True)
class PolicyContext:
    attention_scores: torch.Tensor | None = None
    special_token_mask: torch.Tensor | None = None


class CompressionPolicy(ABC):
    name: str

    @abstractmethod
    def compress(self, past_key_values: KVCacheTensor, context: PolicyContext | None = None) -> CompressionOutcome:
        raise NotImplementedError
