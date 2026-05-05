from __future__ import annotations

import math

import torch


def continuation_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq_len, vocab]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq_len]")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return float(loss.item())


def perplexity_from_nll(nll: float) -> float:
    return float(math.exp(nll))
