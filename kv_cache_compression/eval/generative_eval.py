"""
Generative Evaluation Mode for KV-Cache Compression.

Unlike the NLL-only mode (which measures how likely the *gold* answer is),
generative mode actually runs the model to *produce* text after compression
and scores the generated output against reference answers.

This gives a more realistic picture of quality degradation because:
  - NLL measures the probability of one specific answer token sequence
  - Generative mode measures whether the model can produce *a* correct answer
    (which may differ in phrasing but still be semantically correct)

Supported tasks
---------------
- GenerativeQAEval       : generate answer to a question, score with F1/ROUGE/EM
- GenerativeSummaryEval  : generate summary of an article, score with ROUGE/faithfulness
- GenerativeCodeEval     : generate code completion, score with syntax/identifier/ROUGE-L

All evaluators share a common GenerationConfig for sampling parameters.

Usage
-----
    from kv_cache_compression.eval.generative_eval import (
        GenerativeQAEval, GenerativeSummaryEval, GenerativeCodeEval,
        GenerationConfig, GenerativeResult,
    )
    from kv_cache_compression.cache.advanced_policies import SinkTokenPolicy

    policy = SinkTokenPolicy(num_sink_tokens=4, keep_last_tokens=512)
    cfg    = GenerationConfig(max_new_tokens=64, temperature=0.1)

    qa_eval = GenerativeQAEval(model, tokenizer, gen_config=cfg)
    results = qa_eval.run_all([policy], n_samples=5)
    qa_eval.print_summary(results)
"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Callable

import torch

from kv_cache_compression.cache.kv_cache import KVCacheInspector, to_legacy_tuple
from kv_cache_compression.cache.policies import CompressionPolicy, PolicyContext
from kv_cache_compression.eval.benchmark import _key_norm_scores
from kv_cache_compression.eval.long_qa_eval import get_builtin_samples, LongQASample
from kv_cache_compression.eval.summarization_eval import get_builtin_summaries, SummarizationSample, faithfulness_score
from kv_cache_compression.eval.code_completion_eval import get_builtin_code_samples, CodeCompletionSample, identifier_overlap, _is_valid_python, _edit_distance_ratio
from kv_cache_compression.eval.metrics_eval import token_f1, rouge_scores, exact_match_normalized, needle_recall
from kv_cache_compression.eval.perplexity_eval import perplexity_from_nll


# ══════════════════════════════════════════════════════════════════════════════
# Generation config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationConfig:
    """
    Sampling parameters for generative evaluation.

    Parameters
    ----------
    max_new_tokens  : maximum tokens to generate
    temperature     : sampling temperature (0 = greedy, 1 = default, <1 = sharp)
    top_p           : nucleus sampling probability (1.0 = off)
    top_k           : top-k sampling (0 = off)
    do_sample       : if False, use greedy decoding regardless of temperature
    stop_strings    : list of strings that stop generation early
    """
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 0
    do_sample: bool = True
    stop_strings: list[str] = field(default_factory=lambda: ["\n\n", "Question:", "[Question]", "</s>"])


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerativeResult:
    """Result for one (policy, sample) generative evaluation."""
    policy_name: str
    sample_id: str
    task: str
    generated_text: str
    reference: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    prompt_seconds: float
    compression_seconds: float
    generation_seconds: float
    # Metrics (filled per-task)
    exact_match: float = 0.0
    token_f1: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    needle_recall: float = 0.0
    faithfulness: float = 0.0
    syntax_valid: float = 0.0
    identifier_overlap: float = 0.0
    edit_distance_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Core generation engine
# ══════════════════════════════════════════════════════════════════════════════

class GenerativeEngine:
    """
    Handles the encode → compress → decode generation pipeline.

    Steps
    -----
    1. Tokenise and encode the prompt (prefill).
    2. Apply the compression policy to the KV cache.
    3. Auto-regressively generate new tokens from the compressed cache.
    4. Return the generated text + timing stats.
    """

    def __init__(self, model, tokenizer, gen_config: GenerationConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config or GenerationConfig()
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        policy: CompressionPolicy,
    ) -> tuple[str, dict]:
        """
        Encode prompt, compress KV cache, generate continuation.

        Returns
        -------
        (generated_text, stats_dict)
        stats_dict keys: original_tokens, compressed_tokens, compression_ratio,
                         prompt_seconds, compression_seconds, generation_seconds
        """
        cfg = self.gen_config

        # ── 1. Prefill ──
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self._max_model_len())
        input_ids = inputs["input_ids"].to(self.device)

        t0 = time.perf_counter()
        outputs = self.model(input_ids=input_ids, use_cache=True, output_attentions=False)
        prompt_secs = time.perf_counter() - t0

        past_kv = to_legacy_tuple(outputs.past_key_values)
        original_tokens = KVCacheInspector.sequence_length(past_kv)

        # Compute attention scores (key-norm fallback)
        attn_scores = _key_norm_scores(past_kv)
        context = PolicyContext(attention_scores=attn_scores)

        # ── 2. Compress ──
        t1 = time.perf_counter()
        outcome = policy.compress(past_kv, context=context)
        compress_secs = time.perf_counter() - t1
        past_kv = outcome.past_key_values
        compressed_tokens = outcome.compressed_tokens

        # ── 3. Decode ──
        t2 = time.perf_counter()
        generated_ids: list[int] = []
        next_token_id = outputs.logits[:, -1:, :].argmax(dim=-1)  # greedy first step

        for _ in range(cfg.max_new_tokens):
            token_id = int(next_token_id.item())
            # Check EOS
            if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)

            # Check stop strings
            partial = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(stop in partial for stop in cfg.stop_strings):
                # Trim to stop string
                for stop in cfg.stop_strings:
                    if stop in partial:
                        partial = partial[:partial.index(stop)]
                generated_ids = self.tokenizer.encode(partial, add_special_tokens=False)
                break

            # Next forward pass
            step_out = self.model(
                input_ids=next_token_id,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = to_legacy_tuple(step_out.past_key_values)

            logits = step_out.logits[:, -1, :]
            if not cfg.do_sample or cfg.temperature < 1e-6:
                next_token_id = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / max(cfg.temperature, 1e-6)
                probs  = torch.softmax(scaled, dim=-1)
                if cfg.top_k > 0:
                    topk_vals, _ = probs.topk(cfg.top_k, dim=-1)
                    probs[probs < topk_vals[..., -1:]] = 0.0
                    probs /= probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                if cfg.top_p < 1.0:
                    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                    cum = sorted_probs.cumsum(dim=-1)
                    remove = cum - sorted_probs > cfg.top_p
                    sorted_probs[remove] = 0.0
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                    probs /= probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                next_token_id = torch.multinomial(probs, num_samples=1)

        generation_secs = time.perf_counter() - t2
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(compressed_tokens / max(original_tokens, 1), 4),
            "prompt_seconds": round(prompt_secs, 4),
            "compression_seconds": round(compress_secs, 6),
            "generation_seconds": round(generation_secs, 4),
        }
        return generated_text, stats

    def _max_model_len(self) -> int:
        cfg = getattr(self.model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
        return 2048


# ══════════════════════════════════════════════════════════════════════════════
# Task-specific generative evaluators
# ══════════════════════════════════════════════════════════════════════════════

class GenerativeQAEval:
    """
    Generative Long-Form QA evaluation.

    Generates an answer from the compressed KV cache and scores it against
    the gold reference answer using F1, ROUGE, exact match, and needle recall.
    """

    def __init__(self, model, tokenizer, *, gen_config: GenerationConfig | None = None, verbose: bool = True) -> None:
        self.engine  = GenerativeEngine(model, tokenizer, gen_config)
        self.verbose = verbose

    def run_sample(self, sample: LongQASample, policy: CompressionPolicy) -> GenerativeResult:
        generated, stats = self.engine.generate(sample.prompt, policy)

        rg  = rouge_scores(generated, sample.answer)
        em  = exact_match_normalized(generated, sample.answer)
        f1  = token_f1(generated, sample.answer)
        nr  = needle_recall(generated, sample.answer)

        return GenerativeResult(
            policy_name=policy.name,
            sample_id=sample.doc_id,
            task="qa",
            generated_text=generated,
            reference=sample.answer,
            exact_match=em,
            token_f1=f1,
            rouge1_f1=rg["rouge1"]["f1"],
            rouge2_f1=rg["rouge2"]["f1"],
            rougeL_f1=rg["rougeL"]["f1"],
            needle_recall=nr,
            **stats,
        )

    def run_all(
        self,
        policies: list[CompressionPolicy],
        samples: list[LongQASample] | None = None,
        n_samples: int = 5,
    ) -> dict[str, list[GenerativeResult]]:
        if samples is None:
            samples = get_builtin_samples(n=n_samples)
        results: dict[str, list[GenerativeResult]] = {}
        for policy in policies:
            if self.verbose:
                print(f"\n[GenQA] policy={policy.name}")
            policy_results = []
            for i, sample in enumerate(samples):
                if self.verbose:
                    print(f"  [{i+1}/{len(samples)}] {sample.doc_id} Q: {sample.question[:55]}...")
                try:
                    r = self.run_sample(sample, policy)
                    policy_results.append(r)
                    if self.verbose:
                        print(f"    gen='{r.generated_text[:60]}...' | F1={r.token_f1:.3f} | recall={r.needle_recall:.1f}")
                except Exception as e:
                    if self.verbose:
                        print(f"    FAILED: {e}")
            results[policy.name] = policy_results
        return results

    def aggregate(self, results: list[GenerativeResult]) -> dict:
        return _agg_results(results, ["compression_ratio", "token_f1", "rouge1_f1", "rougeL_f1", "exact_match", "needle_recall", "generation_seconds"])

    def print_summary(self, all_results: dict[str, list[GenerativeResult]]) -> None:
        _print_summary_table("Generative QA", all_results, self.aggregate,
                             ["compression_ratio", "token_f1", "rouge1_f1", "needle_recall"])


class GenerativeSummaryEval:
    """
    Generative Summarisation evaluation.

    Generates a summary of the article from the compressed KV cache and scores
    it against the reference summary using ROUGE + faithfulness.
    """

    def __init__(self, model, tokenizer, *, gen_config: GenerationConfig | None = None, verbose: bool = True) -> None:
        self.engine  = GenerativeEngine(model, tokenizer, gen_config)
        self.verbose = verbose

    def run_sample(self, sample: SummarizationSample, policy: CompressionPolicy) -> GenerativeResult:
        generated, stats = self.engine.generate(sample.prompt, policy)

        rg    = rouge_scores(generated, sample.reference_summary)
        f1    = token_f1(generated, sample.reference_summary)
        faith = faithfulness_score(generated, sample.article)

        return GenerativeResult(
            policy_name=policy.name,
            sample_id=sample.sample_id,
            task="summarization",
            generated_text=generated,
            reference=sample.reference_summary,
            token_f1=f1,
            rouge1_f1=rg["rouge1"]["f1"],
            rouge2_f1=rg["rouge2"]["f1"],
            rougeL_f1=rg["rougeL"]["f1"],
            faithfulness=faith,
            **stats,
        )

    def run_all(
        self,
        policies: list[CompressionPolicy],
        samples: list[SummarizationSample] | None = None,
        n_samples: int = 5,
    ) -> dict[str, list[GenerativeResult]]:
        if samples is None:
            samples = get_builtin_summaries(n=n_samples)
        results: dict[str, list[GenerativeResult]] = {}
        for policy in policies:
            if self.verbose:
                print(f"\n[GenSum] policy={policy.name}")
            policy_results = []
            for i, sample in enumerate(samples):
                if self.verbose:
                    print(f"  [{i+1}/{len(samples)}] {sample.sample_id} ({sample.domain})...")
                try:
                    r = self.run_sample(sample, policy)
                    policy_results.append(r)
                    if self.verbose:
                        print(f"    gen='{r.generated_text[:60]}...' | ROUGE-1={r.rouge1_f1:.3f} | faith={r.faithfulness:.3f}")
                except Exception as e:
                    if self.verbose:
                        print(f"    FAILED: {e}")
            results[policy.name] = policy_results
        return results

    def aggregate(self, results: list[GenerativeResult]) -> dict:
        return _agg_results(results, ["compression_ratio", "rouge1_f1", "rougeL_f1", "faithfulness", "token_f1", "generation_seconds"])

    def print_summary(self, all_results: dict[str, list[GenerativeResult]]) -> None:
        _print_summary_table("Generative Summarisation", all_results, self.aggregate,
                             ["compression_ratio", "rouge1_f1", "rougeL_f1", "faithfulness"])


class GenerativeCodeEval:
    """
    Generative Code Completion evaluation.

    Generates the code completion from the compressed KV cache and scores it
    with syntax validity, identifier overlap, ROUGE-L, and edit distance.
    """

    def __init__(self, model, tokenizer, *, gen_config: GenerationConfig | None = None, verbose: bool = True) -> None:
        cfg = gen_config or GenerationConfig(
            max_new_tokens=128,
            temperature=0.05,        # low temp for code — want deterministic completions
            stop_strings=["\ndef ", "\nclass ", "\n\n\n"],
        )
        self.engine  = GenerativeEngine(model, tokenizer, cfg)
        self.verbose = verbose

    def run_sample(self, sample: CodeCompletionSample, policy: CompressionPolicy) -> GenerativeResult:
        generated, stats = self.engine.generate(sample.full_prompt, policy)

        rg    = rouge_scores(generated, sample.reference)
        f1    = token_f1(generated, sample.reference)
        em    = exact_match_normalized(generated, sample.reference)
        iov   = identifier_overlap(generated, sample.reference)
        syn   = float(_is_valid_python(sample.context + sample.prompt + generated))
        edr   = _edit_distance_ratio(generated, sample.reference)

        return GenerativeResult(
            policy_name=policy.name,
            sample_id=sample.sample_id,
            task="code",
            generated_text=generated,
            reference=sample.reference,
            exact_match=em,
            token_f1=f1,
            rouge1_f1=rg["rouge1"]["f1"],
            rouge2_f1=rg["rouge2"]["f1"],
            rougeL_f1=rg["rougeL"]["f1"],
            syntax_valid=syn,
            identifier_overlap=iov,
            edit_distance_ratio=edr,
            **stats,
        )

    def run_all(
        self,
        policies: list[CompressionPolicy],
        samples: list[CodeCompletionSample] | None = None,
        n_samples: int = 5,
    ) -> dict[str, list[GenerativeResult]]:
        if samples is None:
            samples = get_builtin_code_samples(n=n_samples)
        results: dict[str, list[GenerativeResult]] = {}
        for policy in policies:
            if self.verbose:
                print(f"\n[GenCode] policy={policy.name}")
            policy_results = []
            for i, sample in enumerate(samples):
                if self.verbose:
                    print(f"  [{i+1}/{len(samples)}] {sample.sample_id} ({sample.category})...")
                try:
                    r = self.run_sample(sample, policy)
                    policy_results.append(r)
                    if self.verbose:
                        print(f"    gen='{r.generated_text[:60]}...' | syn={r.syntax_valid:.0f} | id_ov={r.identifier_overlap:.3f}")
                except Exception as e:
                    if self.verbose:
                        print(f"    FAILED: {e}")
            results[policy.name] = policy_results
        return results

    def aggregate(self, results: list[GenerativeResult]) -> dict:
        return _agg_results(results, ["compression_ratio", "syntax_valid", "identifier_overlap", "rougeL_f1", "edit_distance_ratio", "generation_seconds"])

    def print_summary(self, all_results: dict[str, list[GenerativeResult]]) -> None:
        _print_summary_table("Generative Code Completion", all_results, self.aggregate,
                             ["compression_ratio", "syntax_valid", "identifier_overlap", "rougeL_f1"])


# ══════════════════════════════════════════════════════════════════════════════
# Unified generative runner
# ══════════════════════════════════════════════════════════════════════════════

class GenerativeTaskRunner:
    """
    Runs all three generative evaluation tasks across multiple policies.

    Parameters
    ----------
    model, tokenizer : loaded HuggingFace model + tokenizer
    gen_config       : GenerationConfig for all evaluators
    tasks            : subset of ["qa", "summarization", "code"]
    n_samples        : samples per task
    verbose          : print progress
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        gen_config: GenerationConfig | None = None,
        tasks: list[str] | None = None,
        n_samples: int = 5,
        verbose: bool = True,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config or GenerationConfig()
        self.tasks     = tasks or ["qa", "summarization", "code"]
        self.n_samples = n_samples
        self.verbose   = verbose

        self._qa_eval   = GenerativeQAEval(model, tokenizer, gen_config=self.gen_config, verbose=verbose)
        self._sum_eval  = GenerativeSummaryEval(model, tokenizer, gen_config=self.gen_config, verbose=verbose)
        self._code_eval = GenerativeCodeEval(model, tokenizer, verbose=verbose)

    def run_all(self, policies: list[CompressionPolicy]) -> dict[str, dict[str, list[GenerativeResult]]]:
        """
        Run all tasks across all policies.

        Returns
        -------
        {task_name: {policy_name: [GenerativeResult, ...]}}
        """
        all_results: dict[str, dict] = {}

        if "qa" in self.tasks:
            print(f"\n{'─'*60}\n  Generative QA\n{'─'*60}")
            all_results["qa"] = self._qa_eval.run_all(policies, n_samples=self.n_samples)
            self._qa_eval.print_summary(all_results["qa"])

        if "summarization" in self.tasks:
            print(f"\n{'─'*60}\n  Generative Summarisation\n{'─'*60}")
            all_results["summarization"] = self._sum_eval.run_all(policies, n_samples=self.n_samples)
            self._sum_eval.print_summary(all_results["summarization"])

        if "code" in self.tasks:
            print(f"\n{'─'*60}\n  Generative Code Completion\n{'─'*60}")
            all_results["code"] = self._code_eval.run_all(policies, n_samples=self.n_samples)
            self._code_eval.print_summary(all_results["code"])

        return all_results

    def export_json(self, all_results: dict, path: str) -> None:
        import json
        from pathlib import Path
        flat: dict = {}
        for task, task_results in all_results.items():
            flat[task] = {
                pol: [r.to_dict() for r in results]
                for pol, results in task_results.items()
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(flat, f, indent=2, default=str)
        print(f"[generative_eval] Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _agg_results(results: list[GenerativeResult], metrics: list[str]) -> dict:
    if not results:
        return {}

    def _stats(vals):
        clean = [v for v in vals if v is not None]
        if not clean:
            return {"mean": None, "std": None}
        return {
            "mean": round(statistics.mean(clean), 4),
            "std":  round(statistics.stdev(clean) if len(clean) > 1 else 0.0, 4),
        }

    out = {"n": len(results)}
    for m in metrics:
        vals = [getattr(r, m, None) for r in results]
        out[m] = _stats(vals)
    return out


def _print_summary_table(
    title: str,
    all_results: dict[str, list[GenerativeResult]],
    agg_fn: Callable,
    metrics: list[str],
) -> None:
    w = max(len(m) for m in metrics)
    header = f"  {'Policy':<22}" + "".join(f"  {m[:10]:>10}" for m in metrics)
    print(f"\n{'═'*len(header)}")
    print(f"  {title}")
    print(f"{'═'*len(header)}")
    print(header)
    print(f"  {'─'*22}" + "──────────" * len(metrics))
    for pol_name, results in all_results.items():
        agg = agg_fn(results)
        row = f"  {pol_name:<22}"
        for m in metrics:
            val = agg.get(m, {}).get("mean")
            row += f"  {val:>10.3f}" if val is not None else f"  {'N/A':>10}"
        print(row)
    print(f"{'═'*len(header)}")
