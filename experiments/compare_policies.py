"""
Experiment: compare recency, attention, and hybrid compression policies.

Usage:
    python experiments/compare_policies.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python experiments/compare_policies.py --config experiments/configs/default_experiment.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_cache_compression.cache.cluster import HybridClusterPolicy
from kv_cache_compression.cache.prune import AttentionRetentionPolicy, RecencyWindowPolicy
from kv_cache_compression.eval.benchmark import BenchmarkRunner, BenchmarkSample
from kv_cache_compression.eval.needle_eval import make_needle_haystack_sample
from kv_cache_compression.eval.long_qa_eval import build_long_qa_sample
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.utils.profiling import format_bytes
from kv_cache_compression.eval.perplexity_eval import perplexity_from_nll


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare KV-cache compression policies")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--config", default=None, help="Path to JSON config (overrides CLI args)")
    p.add_argument("--keep-last", type=int, default=512)
    p.add_argument("--keep-top", type=int, default=128)
    p.add_argument("--clusters", type=int, default=64)
    p.add_argument("--needle", default="alpha-7319")
    p.add_argument("--haystack-repeats", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-json", default=None, help="Save results to this JSON file")
    return p


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_policies(keep_last: int, keep_top: int, clusters: int) -> list:
    return [
        ("no_compression", None),
        ("recency_window", RecencyWindowPolicy(keep_last_tokens=keep_last)),
        ("attention_retention", AttentionRetentionPolicy(keep_last_tokens=keep_last, keep_top_tokens=keep_top)),
        ("hybrid_cluster", HybridClusterPolicy(keep_last_tokens=keep_last, keep_top_tokens=keep_top, cluster_tokens=clusters)),
    ]


def run_experiment(runner: BenchmarkRunner, sample: BenchmarkSample, policies: list) -> list[dict]:
    results = []
    for policy_label, policy in policies:
        print(f"  → Running policy: {policy_label} ...", flush=True)
        if policy is None:
            # No-compression baseline: run with no policy
            from kv_cache_compression.cache.prune import RecencyWindowPolicy as _R
            # Use a huge keep-last so nothing is pruned (baseline)
            baseline_policy = _R(keep_last_tokens=999_999)
            summary = runner.run(sample, baseline_policy)
            summary_dict = summary.to_dict()
            summary_dict["policy_name"] = "no_compression"
        else:
            summary = runner.run(sample, policy)
            summary_dict = summary.to_dict()

        # Augment with human-readable fields
        summary_dict["original_bytes_human"] = format_bytes(summary_dict["original_bytes"])
        summary_dict["compressed_bytes_human"] = format_bytes(summary_dict["compressed_bytes"])
        if summary_dict["original_tokens"] > 0:
            summary_dict["compression_ratio"] = round(
                summary_dict["compressed_tokens"] / summary_dict["original_tokens"], 4
            )
            summary_dict["tokens_saved_pct"] = round(
                (1 - summary_dict["compressed_tokens"] / summary_dict["original_tokens"]) * 100, 2
            )
        else:
            summary_dict["compression_ratio"] = 1.0
            summary_dict["tokens_saved_pct"] = 0.0

        nll = summary_dict.get("continuation_nll")
        summary_dict["perplexity"] = round(perplexity_from_nll(nll), 4) if nll is not None else None

        results.append(summary_dict)
        print(f"     tokens: {summary_dict['original_tokens']} → {summary_dict['compressed_tokens']} "
              f"({summary_dict['tokens_saved_pct']}% saved) | "
              f"NLL: {nll:.4f}" if nll else f"     tokens: {summary_dict['original_tokens']} → {summary_dict['compressed_tokens']}")
    return results


def print_table(results: list[dict]) -> None:
    header = f"{'Policy':<22} {'Orig Tok':>9} {'Comp Tok':>9} {'Saved%':>8} {'Orig Mem':>10} {'Comp Mem':>10} {'NLL':>8} {'PPL':>9} {'t_prompt':>9} {'t_compress':>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        nll = r.get("continuation_nll")
        ppl = r.get("perplexity")
        print(
            f"{r['policy_name']:<22} "
            f"{r['original_tokens']:>9} "
            f"{r['compressed_tokens']:>9} "
            f"{r['tokens_saved_pct']:>7.1f}% "
            f"{r['original_bytes_human']:>10} "
            f"{r['compressed_bytes_human']:>10} "
            f"{nll:>8.4f}" if nll is not None else f"{'N/A':>8} "
            f"{ppl:>9.2f}" if ppl is not None else f"{'N/A':>9} "
            f"{r['prompt_seconds']:>8.3f}s "
            f"{r['compression_seconds']:>10.4f}s"
        )
    print("=" * len(header))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    model_name   = cfg.get("model", args.model)
    keep_last    = cfg.get("keep_last", args.keep_last)
    keep_top     = cfg.get("keep_top", args.keep_top)
    clusters     = cfg.get("clusters", args.clusters)
    needle       = cfg.get("needle", args.needle)
    repeats      = cfg.get("haystack_repeats", args.haystack_repeats)
    dtype        = cfg.get("dtype", args.dtype)
    device_map   = cfg.get("device_map", args.device_map)
    trust_remote = cfg.get("trust_remote_code", args.trust_remote_code)

    print(f"Loading model: {model_name} (dtype={dtype}, device_map={device_map})")
    tokenizer, model = load_causal_lm(
        model_name,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote,
    )

    needle_data = make_needle_haystack_sample(needle, repeats=repeats)
    sample = BenchmarkSample(
        prompt=needle_data.prompt,
        continuation=needle_data.continuation,
        answer=needle_data.answer,
    )

    print(f"\nBenchmark sample: needle='{needle}', haystack_repeats={repeats}")
    print(f"Prompt length ≈ {len(needle_data.prompt.split())} words\n")

    runner = BenchmarkRunner(model, tokenizer)
    policies = build_policies(keep_last, keep_top, clusters)

    all_results = {"needle_haystack": []}

    print("─── Needle-in-Haystack benchmark ───")
    results = run_experiment(runner, sample, policies)
    all_results["needle_haystack"] = results

    # Also run a short long-QA sample to show multi-task support
    print("\n─── Long-QA benchmark ───")
    long_context = " ".join(f"sentence_{i} contains information about topic_{i % 10}." for i in range(200))
    qa_sample_data = build_long_qa_sample(
        context=long_context,
        question="What topic does sentence_42 relate to?",
        answer="topic_2",
    )
    qa_sample = BenchmarkSample(
        prompt=qa_sample_data.prompt,
        continuation=qa_sample_data.continuation,
        answer=qa_sample_data.answer,
    )
    qa_results = run_experiment(runner, qa_sample, policies)
    all_results["long_qa"] = qa_results

    # Print summary tables
    print("\n\n━━━ Results: Needle-in-Haystack ━━━")
    print_table(all_results["needle_haystack"])

    print("\n\n━━━ Results: Long-QA ━━━")
    print_table(all_results["long_qa"])

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
