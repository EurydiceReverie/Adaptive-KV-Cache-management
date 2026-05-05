from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from kv_cache_compression.cache.kv_cache import KVCacheInspector, to_legacy_tuple
from kv_cache_compression.cache.policies import CompressionPolicy, PolicyContext
from kv_cache_compression.cache.prune import aggregate_attention_scores
from kv_cache_compression.utils.profiling import timed


@dataclass(slots=True)
class BenchmarkSample:
    prompt: str
    continuation: str
    answer: str | None = None


@dataclass(slots=True)
class BenchmarkSummary:
    policy_name: str
    original_tokens: int
    compressed_tokens: int
    original_bytes: int
    compressed_bytes: int
    prompt_seconds: float
    compression_seconds: float
    continuation_nll: float | None
    metadata: dict[str, float | int | str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _key_norm_scores(past_key_values: tuple) -> torch.Tensor | None:
    """
    Fallback token importance scores when output_attentions is unavailable.
    Uses the L2 norm of key vectors averaged across layers and heads as a
    proxy for token importance — tokens with larger key norms tend to receive
    more attention (empirically observed in several papers).

    Expects past_key_values as tuple of (key, value) pairs — use to_legacy_tuple() first.
    """
    if not past_key_values:
        return None
    layer_scores = []
    for layer_idx, layer in enumerate(past_key_values):
        if not (isinstance(layer, (tuple, list)) and len(layer) == 2):
            raise ValueError(
                f"[_key_norm_scores] Layer {layer_idx} is not a (key, value) pair: "
                f"type={type(layer)}, len={len(layer) if hasattr(layer, '__len__') else 'N/A'}"
            )
        key, _value = layer
        # key: [batch, heads, seq_len, head_dim]
        norms = key.norm(dim=-1)          # [batch, heads, seq_len]
        score = norms.mean(dim=1)         # [batch, seq_len]
        layer_scores.append(score)
    stacked = torch.stack(layer_scores, dim=0).mean(dim=0)  # [batch, seq_len]
    return stacked


class BenchmarkRunner:
    def __init__(self, model, tokenizer, *, device: str | torch.device | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else self._resolve_device()

    def _resolve_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        special_ids = [token_id for token_id in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id] if token_id is not None]
        if not special_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_ids:
            mask |= input_ids == token_id
        return mask

    def _get_max_length(self) -> int:
        """Safely retrieve the model's maximum sequence length."""
        cfg = getattr(self.model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        return 2048  # safe default

    def _truncate_inputs(self, inputs: dict) -> dict:
        """Truncate tokenized inputs to the model's max sequence length."""
        max_len = self._get_max_length()
        seq_len = inputs["input_ids"].shape[1]
        if seq_len > max_len:
            print(f"[benchmark] Prompt truncated from {seq_len} → {max_len} tokens (model limit).")
            return {k: v[:, -max_len:] for k, v in inputs.items()}
        return inputs

    def _output_attentions_supported(self) -> bool:
        """Check if the model supports output_attentions (not all attn impls do)."""
        attn_impl = getattr(getattr(self.model, "config", None), "_attn_implementation", "eager")
        return attn_impl in ("eager", "flash_attention_2")

    @torch.inference_mode()
    def run(self, sample: BenchmarkSample, policy: CompressionPolicy) -> BenchmarkSummary:
        prompt_inputs = self.tokenizer(sample.prompt, return_tensors="pt", truncation=False)
        prompt_inputs = {key: value.to(self.device) for key, value in prompt_inputs.items()}
        prompt_inputs = self._truncate_inputs(prompt_inputs)

        use_attentions = self._output_attentions_supported()

        with timed() as prompt_timer:
            outputs = self.model(**prompt_inputs, use_cache=True, output_attentions=use_attentions)

        # Aggregate attention scores if available, else fall back to key-norm scoring
        if use_attentions and outputs.attentions:
            attention_scores = aggregate_attention_scores(outputs.attentions)
        else:
            attention_scores = _key_norm_scores(to_legacy_tuple(outputs.past_key_values))

        context = PolicyContext(
            attention_scores=attention_scores,
            special_token_mask=self._special_token_mask(prompt_inputs["input_ids"]),
        )

        original_cache = to_legacy_tuple(outputs.past_key_values)
        original_tokens = KVCacheInspector.sequence_length(original_cache)
        original_bytes = KVCacheInspector.total_bytes(original_cache)

        with timed() as compression_timer:
            outcome = policy.compress(original_cache, context=context)

        compressed_cache = outcome.past_key_values
        compressed_bytes = KVCacheInspector.total_bytes(compressed_cache)
        continuation_nll = self._teacher_forced_continuation_nll(sample.continuation, compressed_cache)

        return BenchmarkSummary(
            policy_name=outcome.policy_name,
            original_tokens=original_tokens,
            compressed_tokens=outcome.compressed_tokens,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            prompt_seconds=prompt_timer.seconds,
            compression_seconds=compression_timer.seconds,
            continuation_nll=continuation_nll,
            metadata=outcome.metadata,
        )

    @torch.inference_mode()
    def _teacher_forced_continuation_nll(self, continuation: str, past_key_values) -> float | None:
        continuation = continuation or ""
        continuation_inputs = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)
        if continuation_inputs["input_ids"].numel() == 0:
            return None

        input_ids = continuation_inputs["input_ids"].to(self.device)
        total_nll = 0.0
        token_count = 0

        # Convert compressed cache to legacy tuple so the model accepts it cleanly.
        # We re-convert each step because the model may return a DynamicCache again.
        current_past = past_key_values

        for position in range(input_ids.shape[1]):
            current_token = input_ids[:, position : position + 1]
            try:
                outputs = self.model(input_ids=current_token, past_key_values=current_past, use_cache=True)
            except Exception:
                # If the model rejects the tuple cache format, let it build its own
                outputs = self.model(input_ids=current_token, past_key_values=None, use_cache=True)
            current_past = to_legacy_tuple(outputs.past_key_values)
            log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
            target_index = current_token.squeeze(0)
            total_nll += float(-log_probs[0, target_index].item())
            token_count += 1

        return total_nll / max(token_count, 1)
