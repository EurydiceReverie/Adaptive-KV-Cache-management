from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path

from kv_cache_compression.cache.cluster import HybridClusterPolicy
from kv_cache_compression.cache.layerwise import LayerwiseHybridPolicy
from kv_cache_compression.cache.prune import AttentionRetentionPolicy, RecencyWindowPolicy
from kv_cache_compression.cache.advanced_policies import (
    SinkTokenPolicy, HeavyHitterPolicy, QuantizedCachePolicy, ScissorHandsPolicy
)
from kv_cache_compression.cache.learned_memory import LearnedMemoryPolicy
from kv_cache_compression.cache.scheduler import make_scheduler
from kv_cache_compression.cache.streaming import StreamingCompressor, generate_with_compression
from kv_cache_compression.eval.benchmark import BenchmarkRunner, BenchmarkSample
from kv_cache_compression.eval.needle_eval import make_needle_haystack_sample
from kv_cache_compression.eval.metrics_eval import token_f1, rouge_scores, needle_recall
from kv_cache_compression.eval.perplexity_eval import perplexity_from_nll
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.utils.profiling import format_bytes
from kv_cache_compression.utils.plotting import (
    plot_compression_vs_quality,
    plot_memory_vs_quality,
    plot_latency_breakdown,
    plot_pareto_frontier,
    plot_radar_chart,
    plot_budget_schedule,
)


POLICY_CHOICES = [
    "recency", "attention", "hybrid", "layerwise",
    "sink", "h2o", "quant", "scissor",
    "learned",   # Phase 5: learned synthetic memory tokens
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KV-Cache Compression — Advanced Research CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name or path")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Policy
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="hybrid",
                        help="Compression policy to use")
    parser.add_argument("--keep-last", type=int, default=512,
                        help="Recency window size (tokens)")
    parser.add_argument("--keep-top", type=int, default=128,
                        help="Top-k attention tokens to keep")
    parser.add_argument("--clusters", type=int, default=64,
                        help="Number of cluster representatives (hybrid policy)")
    parser.add_argument("--sink-tokens", type=int, default=4,
                        help="Number of sink tokens (sink/h2o/scissor policies)")
    parser.add_argument("--h2o-budget", type=int, default=512,
                        help="Total token budget for H2O heavy-hitter policy")
    parser.add_argument("--h2o-decay", type=float, default=0.9,
                        help="Exponential decay for H2O accumulated scores")

    # Phase 5 — Learned memory token options
    parser.add_argument("--n-memory", type=int, default=32,
                        help="Number of synthetic memory tokens (learned policy)")
    parser.add_argument("--memory-optim-steps", type=int, default=50,
                        help="Gradient optimisation steps for learned memory policy")
    parser.add_argument("--memory-lr", type=float, default=1e-2,
                        help="Adam learning rate for memory token optimisation")
    parser.add_argument("--memory-keep-last", type=int, default=64,
                        help="Recency anchor: real tokens always kept alongside memory tokens")

    # Scheduler
    parser.add_argument("--scheduler", choices=["none", "linear", "exponential", "memory"],
                        default="none", help="Adaptive budget scheduler")
    parser.add_argument("--sched-start-len", type=int, default=512)
    parser.add_argument("--sched-end-len", type=int, default=4096)
    parser.add_argument("--sched-start-ratio", type=float, default=0.9)
    parser.add_argument("--sched-end-ratio", type=float, default=0.25)

    # Benchmark task
    parser.add_argument("--needle", default="alpha-7319")
    parser.add_argument("--haystack-repeats", type=int, default=256)

    # Streaming generation
    parser.add_argument("--stream", action="store_true",
                        help="Run streaming generation with periodic compression")
    parser.add_argument("--stream-compress-every", type=int, default=64,
                        help="Compress every N tokens during streaming generation")
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="Maximum new tokens to generate in streaming mode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Task selection
    parser.add_argument("--task", choices=["benchmark", "qa", "summarization", "code", "all"],
                        default="benchmark",
                        help=(
                            "Evaluation task to run: "
                            "'benchmark'=needle-in-haystack (default), "
                            "'qa'=long-form QA, "
                            "'summarization'=article summarisation, "
                            "'code'=Python code completion, "
                            "'all'=run all three"
                        ))
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples per task (qa/sum/code)")

    # Output
    parser.add_argument("--output-json", default=None,
                        help="Save benchmark results to JSON file")
    parser.add_argument("--export-csv", default=None,
                        help="Export results to CSV file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save comparison charts")
    parser.add_argument("--plot-dir", default="experiments/results/plots",
                        help="Directory to save plots")
    parser.add_argument("--plot-budget-schedule", action="store_true",
                        help="Plot the budget schedule curve (requires --scheduler != none)")

    return parser


def make_policy(args: argparse.Namespace):
    """Instantiate the compression policy from CLI args."""
    p = args.policy
    if p == "recency":
        return RecencyWindowPolicy(keep_last_tokens=args.keep_last)
    if p == "attention":
        return AttentionRetentionPolicy(keep_last_tokens=args.keep_last, keep_top_tokens=args.keep_top)
    if p == "layerwise":
        return LayerwiseHybridPolicy(keep_last_tokens=args.keep_last, keep_top_tokens=args.keep_top)
    if p == "sink":
        return SinkTokenPolicy(num_sink_tokens=args.sink_tokens, keep_last_tokens=args.keep_last)
    if p == "h2o":
        return HeavyHitterPolicy(
            budget=args.h2o_budget,
            num_sink_tokens=args.sink_tokens,
            decay=args.h2o_decay,
        )
    if p == "quant":
        return QuantizedCachePolicy(keep_last_tokens=args.keep_last)
    if p == "scissor":
        return ScissorHandsPolicy(
            num_sink_tokens=args.sink_tokens,
            keep_top_tokens=args.keep_top,
            keep_last_tokens=args.keep_last,
        )
    if p == "learned":
        return LearnedMemoryPolicy(
            n_memory=args.n_memory,
            optim_steps=args.memory_optim_steps,
            lr=args.memory_lr,
            always_keep_last=args.memory_keep_last,
        )
    # default: hybrid cluster
    return HybridClusterPolicy(
        keep_last_tokens=args.keep_last,
        keep_top_tokens=args.keep_top,
        cluster_tokens=args.clusters,
    )


def _export_csv(results: list[dict], path: str) -> None:
    """Write results list to a CSV file."""
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        meta = r.get("metadata", {})
        for k, v in meta.items():
            row[f"meta_{k}"] = v
        flat_rows.append(row)
    fieldnames = list(flat_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[cli] Results exported → {path}")


def _run_task_mode(args, tokenizer, model, policy) -> int:
    """Run multi-task evaluation (qa / summarization / code / all)."""
    from kv_cache_compression.eval.task_runner import TaskRunner, TaskConfig

    task_map = {
        "qa":            ["qa"],
        "summarization": ["summarization"],
        "code":          ["code"],
        "all":           ["qa", "summarization", "code"],
    }
    tasks = task_map.get(args.task, ["qa", "summarization", "code"])

    config = TaskConfig(
        tasks=tasks,
        n_samples=args.n_samples,
        verbose=True,
    )
    runner = TaskRunner(model, tokenizer, config=config)
    report = runner.run_all([policy])

    runner.export(
        report,
        json_path=args.output_json,
        csv_path=args.export_csv,
    )

    if args.plot:
        # Build flat result list for radar chart
        results = []
        for task, ts in report.task_summaries.items():
            for pol_name, agg in ts.policy_summaries.items():
                row = {"policy_name": f"{pol_name}_{task}"}
                for metric, stats in agg.items():
                    if isinstance(stats, dict):
                        row[metric] = stats.get("mean")
                results.append(row)
        if results:
            plot_radar_chart(
                results,
                title=f"Task Comparison — {policy.name}",
                save_path=f"{args.plot_dir}/radar_{policy.name}.png",
            )
            print(f"[cli] Radar chart saved to {args.plot_dir}/")

    return 0


def _run_streaming_mode(args, tokenizer, model, policy) -> int:
    """Run streaming generation with periodic KV-cache compression."""
    import torch

    scheduler = None
    if args.scheduler != "none":
        scheduler = make_scheduler(
            args.scheduler,
            start_len=args.sched_start_len,
            end_len=args.sched_end_len,
            start_ratio=args.sched_start_ratio,
            end_ratio=args.sched_end_ratio,
        )

    compressor = StreamingCompressor(
        policy=policy,
        compress_every=args.stream_compress_every,
        scheduler=scheduler,
        on_compress=lambda s: print(
            f"  [stream] step={s.step:>4}  "
            f"{s.seq_len_before:>5}→{s.seq_len_after:>5} tokens  "
            f"evicted={s.tokens_evicted:>4}  ratio={s.compression_ratio:.3f}"
        ),
    )

    from kv_cache_compression.eval.needle_eval import make_needle_haystack_sample
    sample_data = make_needle_haystack_sample(args.needle, repeats=args.haystack_repeats)
    prompt = sample_data.prompt

    device = "cpu"
    try:
        import torch
        device = next(model.parameters()).device
    except Exception:
        pass

    print(f"\n[stream] Generating up to {args.max_new_tokens} tokens with streaming compression...")
    print(f"[stream] Policy: {policy.name}  compress_every={args.stream_compress_every}\n")

    generated, summary = generate_with_compression(
        model, tokenizer, prompt, compressor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(f"\n{'─'*60}")
    print(f"Generated text:\n  {generated[:300]}{'...' if len(generated) > 300 else ''}")
    print(f"\nStreaming summary:")
    print(json.dumps(summary, indent=2))

    if args.plot:
        from kv_cache_compression.utils.plotting import plot_streaming_history
        plot_streaming_history(
            compressor.history,
            title=f"Streaming Compression — {policy.name}",
            save_path=f"{args.plot_dir}/streaming_{policy.name}.png",
        )

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading model: {args.model}  (dtype={args.dtype}, device_map={args.device_map})")
    tokenizer, model = load_causal_lm(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    policy = make_policy(args)
    print(f"Policy: {policy.name}")

    # ── Plot budget schedule if requested ──
    if args.plot_budget_schedule and args.scheduler != "none":
        scheduler = make_scheduler(
            args.scheduler,
            start_len=args.sched_start_len,
            end_len=args.sched_end_len,
            start_ratio=args.sched_start_ratio,
            end_ratio=args.sched_end_ratio,
        )
        plot_budget_schedule(
            scheduler,
            title=f"Budget Schedule — {args.scheduler}",
            save_path=f"{args.plot_dir}/budget_schedule_{args.scheduler}.png",
        )

    # ── Streaming generation mode ──
    if args.stream:
        return _run_streaming_mode(args, tokenizer, model, policy)

    # ── Multi-task evaluation mode ──
    if args.task in ("qa", "summarization", "code", "all"):
        return _run_task_mode(args, tokenizer, model, policy)

    # ── Standard single-policy benchmark ──
    sample_data = make_needle_haystack_sample(args.needle, repeats=args.haystack_repeats)
    sample = BenchmarkSample(
        prompt=sample_data.prompt,
        continuation=sample_data.continuation,
        answer=sample_data.answer,
    )

    runner = BenchmarkRunner(model, tokenizer)
    summary = runner.run(sample, policy)

    payload = summary.to_dict()
    payload["original_bytes_human"]   = format_bytes(summary.original_bytes)
    payload["compressed_bytes_human"] = format_bytes(summary.compressed_bytes)

    orig = payload["original_tokens"]
    comp = payload["compressed_tokens"]
    if orig > 0:
        payload["compression_ratio"]  = round(comp / orig, 4)
        payload["tokens_saved_pct"]   = round((1 - comp / orig) * 100, 2)
    else:
        payload["compression_ratio"]  = 1.0
        payload["tokens_saved_pct"]   = 0.0

    nll = payload.get("continuation_nll")
    payload["perplexity"] = round(perplexity_from_nll(nll), 4) if nll is not None else None

    # NLG quality metrics on continuation vs answer
    pred  = sample_data.continuation.strip()
    ref   = sample_data.answer.strip()
    payload["token_f1"]       = token_f1(pred, ref)
    payload["needle_recall"]  = needle_recall(pred, args.needle)
    rg = rouge_scores(pred, ref)
    payload["rouge1_f1"]      = rg["rouge1"]["f1"]
    payload["rouge2_f1"]      = rg["rouge2"]["f1"]
    payload["rougeL_f1"]      = rg["rougeL"]["f1"]

    print(json.dumps(payload, indent=2))

    # ── Save JSON ──
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[cli] Results saved → {args.output_json}")

    # ── Export CSV ──
    if args.export_csv:
        _export_csv([payload], args.export_csv)

    # ── Plots ──
    if args.plot:
        results = [payload]
        plot_compression_vs_quality(
            results,
            title=f"Compression vs Quality — {policy.name}",
            save_path=f"{args.plot_dir}/{policy.name}_compression_quality.png",
        )
        plot_pareto_frontier(
            results,
            save_path=f"{args.plot_dir}/{policy.name}_pareto.png",
        )
        print(f"[cli] Plots saved to {args.plot_dir}/")

    return 0
