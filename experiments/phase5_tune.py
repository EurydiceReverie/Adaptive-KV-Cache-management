"""
Phase 5 Tuning — Grid Search over n_memory × optim_steps
=========================================================
Systematically experiments with different values of:
  - n_memory     : number of synthetic memory tokens (8, 16, 32, 64, 128)
  - optim_steps  : gradient optimisation steps (0, 10, 25, 50, 100)
  - always_keep_last : recency anchor size (16, 32, 64)

For each combination, runs the policy on all prompts and records:
  - Compression ratio (how much memory saved)
  - NLL / Perplexity (quality of preserved information)
  - Compression time (compute cost)
  - Token reduction (how many tokens removed)

Produces:
  - experiments/results/phase5/tuning_results.json   (raw data)
  - experiments/results/phase5/tuning_report.txt     (human readable table)
  - Printed heatmap: n_memory vs optim_steps → quality

Usage:
    # Quick grid (CPU-friendly, 3 prompts)
    python experiments/phase5_tune.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --n-prompts 2 \\
        --output-dir experiments/results/phase5

    # Full grid with local model
    python experiments/phase5_tune.py \\
        --model models/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \\
        --n-prompts 5 \\
        --output-dir experiments/results/phase5

    # Custom grid
    python experiments/phase5_tune.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --n-memory-values 8 16 32 64 \\
        --optim-steps-values 0 10 30 50 \\
        --keep-last-values 32 64 \\
        --n-prompts 3
"""

from __future__ import annotations
import argparse, json, os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.cache.learned_memory import LearnedMemoryPolicy
from kv_cache_compression.eval.benchmark import BenchmarkRunner, BenchmarkSample, _key_norm_scores
from kv_cache_compression.cache.kv_cache import KVCacheInspector, to_legacy_tuple
from kv_cache_compression.cache.prune import aggregate_attention_scores
from kv_cache_compression.cache.policies import PolicyContext

# ── Diverse prompts for tuning (long enough to trigger compression) ───────────
TUNE_PROMPTS = [
    BenchmarkSample(
        prompt=(
            "The KV cache is a fundamental component of autoregressive transformer "
            "inference. During generation, the model computes key and value vectors for "
            "every input token and stores them to avoid recomputation. As generation "
            "proceeds, the cache grows linearly with the number of tokens. For a 32-layer "
            "model with 32 attention heads and head_dim=128, each token adds "
            "32 × 32 × 2 × 128 × 2 = 524,288 bytes = 0.5 MB to the cache. After 2048 "
            "tokens that is 1 GB — just for the cache. KV-cache compression reduces this "
            "by selectively removing or summarising less important tokens. The challenge "
            "is to maximise memory savings while preserving generation quality."
        ),
        continuation="Compression reduces memory while preserving quality.",
        answer="compression"
    ),
    BenchmarkSample(
        prompt=(
            "Neural networks learn representations through gradient descent. The backprop "
            "algorithm computes gradients of a loss function with respect to all parameters "
            "by applying the chain rule layer by layer. Modern deep learning frameworks "
            "like PyTorch use automatic differentiation — they build a computational graph "
            "during the forward pass and traverse it in reverse during the backward pass. "
            "Optimisers like Adam combine momentum (first moment) with adaptive learning "
            "rates (second moment) to speed convergence. Regularisation techniques such as "
            "dropout, weight decay, and batch normalisation prevent overfitting. "
            "Transformer models have billions of parameters trained on trillions of tokens."
        ),
        continuation="Deep learning models are trained using gradient descent.",
        answer="gradient descent"
    ),
    BenchmarkSample(
        prompt=(
            "The Amazon rainforest is the world's largest tropical rainforest, covering "
            "5.5 million square kilometres. It produces 20% of Earth's oxygen and is home "
            "to 10% of all species. Deforestation threatens this ecosystem: 17% has already "
            "been cleared. Scientists warn of a tipping point at 20-25% deforestation beyond "
            "which the forest could irreversibly transition to savannah. The Amazon River "
            "carries 20% of all fresh water discharged into the world's oceans. Indigenous "
            "communities have protected the forest for millennia. Climate change is making "
            "droughts more frequent, stressing trees and increasing fire risk. Conservation "
            "efforts include protected areas, carbon credits, and satellite monitoring."
        ),
        continuation="The Amazon is critical for global climate stability.",
        answer="Amazon"
    ),
    BenchmarkSample(
        prompt=(
            "Python was created by Guido van Rossum in 1991. It is known for its clean "
            "syntax and emphasis on readability. The language supports multiple programming "
            "paradigms including procedural, object-oriented, and functional styles. "
            "Python's package ecosystem (PyPI) has over 400,000 packages. NumPy provides "
            "efficient array operations. Pandas offers data manipulation tools. Matplotlib "
            "enables data visualisation. PyTorch and TensorFlow are the dominant deep "
            "learning frameworks. Python consistently ranks as the most popular programming "
            "language globally. Its interpreted nature makes it slower than C++ but much "
            "faster to develop with. The GIL (Global Interpreter Lock) limits true "
            "multi-threading but multiprocessing and async IO provide concurrency options."
        ),
        continuation="Python is the most popular programming language for AI research.",
        answer="Python"
    ),
    BenchmarkSample(
        prompt=(
            "Quantum mechanics describes physical phenomena at atomic and subatomic scales. "
            "The wave-particle duality states that particles like electrons exhibit both "
            "wave and particle properties depending on how they are observed. Heisenberg's "
            "uncertainty principle states that position and momentum cannot both be known "
            "precisely simultaneously. Schrödinger's equation describes how quantum states "
            "evolve over time. Quantum entanglement — 'spooky action at a distance' as "
            "Einstein called it — links the states of particles regardless of distance. "
            "This non-locality has been confirmed experimentally many times. Quantum "
            "computing exploits superposition and entanglement to perform computations "
            "that are infeasible for classical computers in specific domains."
        ),
        continuation="Quantum mechanics is foundational to modern physics.",
        answer="quantum"
    ),
]


def run_config(
    runner: BenchmarkRunner,
    prompts: list[BenchmarkSample],
    n_memory: int,
    optim_steps: int,
    always_keep_last: int,
) -> dict:
    """Run one grid configuration and return averaged metrics."""
    policy = LearnedMemoryPolicy(
        n_memory=n_memory,
        optim_steps=optim_steps,
        lr=1e-2,
        use_kmeans_init=True,
        always_keep_last=always_keep_last,
    )

    ratios, nlls, times, reductions = [], [], [], []
    errors = []

    for sample in prompts:
        try:
            t0 = time.time()
            summary = runner.run(sample, policy)
            elapsed = time.time() - t0

            ratio = summary.compressed_tokens / max(summary.original_tokens, 1)
            nll = summary.continuation_nll or 0.0
            reduction = summary.original_tokens - summary.compressed_tokens

            ratios.append(ratio)
            nlls.append(nll)
            times.append(elapsed)
            reductions.append(reduction)
        except Exception as e:
            errors.append(str(e))

    def avg(lst): return round(sum(lst) / len(lst), 5) if lst else None

    return {
        "n_memory": n_memory,
        "optim_steps": optim_steps,
        "always_keep_last": always_keep_last,
        "avg_compression_ratio": avg(ratios),
        "avg_nll": avg(nlls),
        "avg_perplexity": round(2 ** avg(nlls), 3) if avg(nlls) else None,
        "avg_time_seconds": avg(times),
        "avg_token_reduction": avg(reductions),
        "n_successful": len(ratios),
        "n_errors": len(errors),
        "errors": errors[:3],  # keep first 3 errors only
    }


def print_heatmap(results: list[dict], metric: str = "avg_nll", title: str = "NLL") -> None:
    """Print ASCII heatmap of metric over n_memory × optim_steps grid."""
    n_memory_vals = sorted(set(r["n_memory"] for r in results))
    optim_vals    = sorted(set(r["optim_steps"] for r in results))

    # Index results
    grid = {}
    for r in results:
        grid[(r["n_memory"], r["optim_steps"])] = r.get(metric)

    all_vals = [v for v in grid.values() if v is not None]
    if not all_vals:
        return
    vmin, vmax = min(all_vals), max(all_vals)
    vrange = vmax - vmin if vmax != vmin else 1.0

    shade = ["░", "▒", "▓", "█"]

    print(f"\n  📊 HEATMAP: {title}  (lower = better for NLL/PPL, higher = better for ratio)")
    print(f"  n_memory \\ optim_steps →")
    header = "  n_mem\\steps  " + "".join(f"{s:>8}" for s in optim_vals)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for nm in n_memory_vals:
        row = f"  {nm:>12}  "
        for os_ in optim_vals:
            val = grid.get((nm, os_))
            if val is None:
                row += "       -"
            else:
                # Normalise 0–1 (for NLL: lower is better → invert)
                norm = (val - vmin) / vrange
                if metric in ("avg_nll", "avg_perplexity", "avg_time_seconds"):
                    # Lower is better — shade dark = bad (high value)
                    idx = min(3, int(norm * 4))
                    cell = shade[idx]
                else:
                    # Higher is better (compression_ratio, token_reduction)
                    idx = min(3, int((1 - norm) * 4))
                    cell = shade[idx]
                row += f"  {val:>5.3f}{cell}"
        print(row)

    print(f"\n  Shade: ░ = good ({'low' if metric in ('avg_nll','avg_perplexity','avg_time_seconds') else 'low'})  "
          f"█ = bad ({'high' if metric in ('avg_nll','avg_perplexity') else 'high'})\n")


def print_tuning_table(results: list[dict]) -> None:
    """Print sorted leaderboard of all configurations."""
    valid = [r for r in results if r["avg_nll"] is not None]
    if not valid:
        print("No valid results to display.")
        return

    # Sort by NLL (lower is better quality)
    sorted_r = sorted(valid, key=lambda x: x["avg_nll"])

    print(f"\n{'═'*90}")
    print(f"  Phase 5 Tuning Results — Sorted by NLL (lower = better quality)")
    print(f"{'─'*90}")
    print(f"  {'RANK':>4}  {'n_mem':>5}  {'steps':>5}  {'keep_last':>9}  "
          f"{'RATIO':>7}  {'NLL':>8}  {'PPL':>8}  {'TIME':>7}  {'TOK↓':>6}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}")

    best_nll = sorted_r[0]["avg_nll"]
    for rank, r in enumerate(sorted_r, 1):
        star = " ⭐" if r["avg_nll"] == best_nll else ""
        ratio_pct = (r["avg_compression_ratio"] or 0) * 100
        print(f"  {rank:>4}  {r['n_memory']:>5}  {r['optim_steps']:>5}  "
              f"{r['always_keep_last']:>9}  "
              f"{ratio_pct:>6.1f}%  {r['avg_nll']:>8.5f}  "
              f"{r['avg_perplexity']:>8.3f}  {r['avg_time_seconds']:>6.2f}s  "
              f"{int(r['avg_token_reduction'] or 0):>6}{star}")

    print(f"{'═'*90}")
    best = sorted_r[0]
    print(f"\n  🏆 BEST CONFIG (lowest NLL):")
    print(f"     n_memory={best['n_memory']}, optim_steps={best['optim_steps']}, "
          f"always_keep_last={best['always_keep_last']}")
    print(f"     NLL={best['avg_nll']:.5f}  PPL={best['avg_perplexity']:.3f}  "
          f"ratio={best['avg_compression_ratio']*100:.1f}%  time={best['avg_time_seconds']:.2f}s")

    # Best compression/quality tradeoff (Pareto: lowest NLL × lowest ratio)
    # Simple proxy: score = nll * compression_ratio (both lower = better)
    for r in valid:
        r["_pareto_score"] = (r["avg_nll"] or 9) * (r["avg_compression_ratio"] or 1)
    pareto_best = min(valid, key=lambda x: x["_pareto_score"])
    print(f"\n  ⚖️  BEST TRADEOFF (quality × compression):")
    print(f"     n_memory={pareto_best['n_memory']}, optim_steps={pareto_best['optim_steps']}, "
          f"always_keep_last={pareto_best['always_keep_last']}")
    print(f"     NLL={pareto_best['avg_nll']:.5f}  PPL={pareto_best['avg_perplexity']:.3f}  "
          f"ratio={pareto_best['avg_compression_ratio']*100:.1f}%  "
          f"time={pareto_best['avg_time_seconds']:.2f}s\n")


def run_tuning(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parse grid values ────────────────────────────────────────────────────
    n_memory_vals    = sorted(set(args.n_memory_values))
    optim_steps_vals = sorted(set(args.optim_steps_values))
    keep_last_vals   = sorted(set(args.keep_last_values))

    total_configs = len(n_memory_vals) * len(optim_steps_vals) * len(keep_last_vals)
    total_runs    = total_configs * min(args.n_prompts, len(TUNE_PROMPTS))

    print("\n" + "═"*65)
    print("  Phase 5 — Hyperparameter Tuning Grid Search")
    print("═"*65)
    print(f"  Model          : {args.model}")
    print(f"  n_memory grid  : {n_memory_vals}")
    print(f"  optim_steps    : {optim_steps_vals}")
    print(f"  keep_last      : {keep_last_vals}")
    print(f"  Grid size      : {total_configs} configs × {min(args.n_prompts, len(TUNE_PROMPTS))} prompts = {total_runs} runs")
    print(f"  Output dir     : {args.output_dir}")
    print("═"*65 + "\n")

    print("⏳ Loading model...")
    t0 = time.time()
    tokenizer, model = load_causal_lm(
        args.model, device_map=args.device_map, dtype=args.dtype
    )
    device = next(model.parameters()).device
    print(f"✅ Model loaded on {device} in {time.time()-t0:.1f}s\n")

    runner = BenchmarkRunner(model, tokenizer)
    prompts = TUNE_PROMPTS[:args.n_prompts]

    all_results = []
    config_num = 0

    for n_mem, n_steps, keep_last in itertools.product(
        n_memory_vals, optim_steps_vals, keep_last_vals
    ):
        config_num += 1
        print(f"  [{config_num:>3}/{total_configs}] n_memory={n_mem:>4}  "
              f"optim_steps={n_steps:>4}  keep_last={keep_last:>4} ...", end="", flush=True)

        t_cfg = time.time()
        result = run_config(runner, prompts, n_mem, n_steps, keep_last)
        elapsed = time.time() - t_cfg

        nll = result["avg_nll"]
        ppl = result["avg_perplexity"]
        ratio = (result["avg_compression_ratio"] or 0) * 100
        print(f"  NLL={nll:.4f}  PPL={ppl:.2f}  ratio={ratio:.1f}%  [{elapsed:.1f}s]")

        all_results.append(result)

        # Save intermediate results (so you can inspect early)
        json_path = os.path.join(args.output_dir, "tuning_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Final report ─────────────────────────────────────────────────────────
    print_tuning_table(all_results)

    if len(n_memory_vals) > 1 and len(optim_steps_vals) > 1:
        # Only show heatmap for the first keep_last value (most informative)
        kl = keep_last_vals[0]
        subset = [r for r in all_results if r["always_keep_last"] == kl]
        print_heatmap(subset, metric="avg_nll",
                      title=f"NLL (keep_last={kl}) — lower=better")
        print_heatmap(subset, metric="avg_compression_ratio",
                      title=f"Compression Ratio (keep_last={kl}) — higher=more compressed")
        print_heatmap(subset, metric="avg_time_seconds",
                      title=f"Time per sample (keep_last={kl}) — lower=faster")

    # Save final JSON + text report
    json_path = os.path.join(args.output_dir, "tuning_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    report_path = os.path.join(args.output_dir, "tuning_report.txt")
    _write_text_report(all_results, report_path, n_memory_vals, optim_steps_vals, keep_last_vals)

    print(f"\n✅ Tuning complete!")
    print(f"   Raw JSON  : {json_path}")
    print(f"   Report    : {report_path}")
    print(f"\n💡 Tip: Use the best config in phase5_compare_all.py with:")
    valid = [r for r in all_results if r["avg_nll"] is not None]
    if valid:
        best = min(valid, key=lambda x: x["avg_nll"])
        print(f"   --n-memory {best['n_memory']} --optim-steps {best['optim_steps']} "
              f"--keep-last-learned {best['always_keep_last']}")


def _write_text_report(results, path, n_mem_vals, step_vals, kl_vals):
    lines = [
        "Phase 5 Hyperparameter Tuning Report",
        "=" * 80,
        f"n_memory values    : {n_mem_vals}",
        f"optim_steps values : {step_vals}",
        f"keep_last values   : {kl_vals}",
        "",
        f"{'RANK':>4}  {'n_mem':>5}  {'steps':>5}  {'keep_last':>9}  "
        f"{'RATIO%':>7}  {'NLL':>8}  {'PPL':>8}  {'TIME':>7}",
        "─" * 70,
    ]
    valid = sorted(
        [r for r in results if r["avg_nll"] is not None],
        key=lambda x: x["avg_nll"]
    )
    for rank, r in enumerate(valid, 1):
        lines.append(
            f"  {rank:>2}  {r['n_memory']:>5}  {r['optim_steps']:>5}  "
            f"{r['always_keep_last']:>9}  "
            f"{(r['avg_compression_ratio'] or 0)*100:>6.1f}%  "
            f"{r['avg_nll']:>8.5f}  {r['avg_perplexity']:>8.3f}  "
            f"{r['avg_time_seconds']:>6.2f}s"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5 hyperparameter tuning: grid search over n_memory × optim_steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HuggingFace model ID or local path (e.g. models/TinyLlama__...)")
    p.add_argument("--n-prompts", type=int, default=3,
                   help="Prompts per config (1-5). More = more reliable but slower.")

    # Grid axes
    p.add_argument("--n-memory-values", nargs="+", type=int,
                   default=[8, 16, 32, 64],
                   help="List of n_memory values to sweep")
    p.add_argument("--optim-steps-values", nargs="+", type=int,
                   default=[0, 10, 30, 50],
                   help="List of optim_steps values to sweep (0 = K-Means init only, no optimisation)")
    p.add_argument("--keep-last-values", nargs="+", type=int,
                   default=[32, 64],
                   help="List of always_keep_last (recency anchor) values to sweep")

    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--output-dir", default="experiments/results/phase5",
                   help="Directory for tuning results")
    return p


if __name__ == "__main__":
    run_tuning(build_parser().parse_args())
