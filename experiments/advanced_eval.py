"""
Advanced Experiment: Compare ALL 8 compression policies across ALL 3 tasks.

Runs Long-form QA, Summarisation, and Code Completion evaluations for each
of the 8 built-in compression policies, exports CSV + JSON results, and
generates comparison plots (radar chart, Pareto frontier, per-task bars).

Usage
-----
    # Quick run (2 samples per task, TinyLlama)
    python experiments/advanced_eval.py

    # Full run with custom model and more samples
    python experiments/advanced_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-samples 10 --plot

    # Run only specific tasks and policies
    python experiments/advanced_eval.py --tasks qa code --policies recency sink h2o --n-samples 5

    # Load from config file
    python experiments/advanced_eval.py --config experiments/configs/default_experiment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_cache_compression.cache.cluster import HybridClusterPolicy
from kv_cache_compression.cache.layerwise import LayerwiseHybridPolicy
from kv_cache_compression.cache.prune import AttentionRetentionPolicy, RecencyWindowPolicy
from kv_cache_compression.cache.advanced_policies import (
    SinkTokenPolicy, HeavyHitterPolicy, QuantizedCachePolicy, ScissorHandsPolicy
)
from kv_cache_compression.eval.task_runner import TaskRunner, TaskConfig
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.utils.profiling import format_bytes


# ══════════════════════════════════════════════════════════════════════════════
# Policy registry
# ══════════════════════════════════════════════════════════════════════════════

def build_all_policies(keep_last: int = 512, keep_top: int = 128, clusters: int = 64) -> dict:
    """Return all 8 built-in compression policies by name."""
    return {
        "recency":      RecencyWindowPolicy(keep_last_tokens=keep_last),
        "attention":    AttentionRetentionPolicy(keep_last_tokens=keep_last, keep_top_tokens=keep_top),
        "hybrid":       HybridClusterPolicy(keep_last_tokens=keep_last, keep_top_tokens=keep_top, cluster_tokens=clusters),
        "layerwise":    LayerwiseHybridPolicy(keep_last_tokens=keep_last, keep_top_tokens=keep_top),
        "sink":         SinkTokenPolicy(num_sink_tokens=4, keep_last_tokens=keep_last),
        "h2o":          HeavyHitterPolicy(budget=keep_last + keep_top, num_sink_tokens=4),
        "quant":        QuantizedCachePolicy(keep_last_tokens=0),
        "scissor":      ScissorHandsPolicy(num_sink_tokens=4, keep_top_tokens=keep_top, keep_last_tokens=keep_last),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced KV-Cache Compression Experiment: All Policies × All Tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--tasks", nargs="+", default=["qa", "summarization", "code"],
                   choices=["qa", "summarization", "code"],
                   help="Which tasks to evaluate")
    p.add_argument("--policies", nargs="+",
                   default=["recency", "attention", "hybrid", "layerwise", "sink", "h2o", "quant", "scissor"],
                   choices=["recency", "attention", "hybrid", "layerwise", "sink", "h2o", "quant", "scissor"],
                   help="Which policies to compare")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of samples per task")

    p.add_argument("--keep-last", type=int, default=512)
    p.add_argument("--keep-top",  type=int, default=128)
    p.add_argument("--clusters",  type=int, default=64)

    p.add_argument("--results-dir", default="experiments/results",
                   help="Directory for output files")
    p.add_argument("--plot", action="store_true", help="Generate comparison plots")
    p.add_argument("--plot-dir", default="experiments/results/plots")
    p.add_argument("--config", default=None, help="JSON config file (overrides CLI args)")
    return p


def load_config_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _plot_all(report, plot_dir: str, tasks: list[str]) -> None:
    """Generate all comparison plots from the task report."""
    from kv_cache_compression.utils.plotting import (
        plot_pareto_frontier, plot_radar_chart, plot_compression_vs_quality
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Build flat result rows: one per (task, policy) with key metrics
    all_rows: list[dict] = []
    for task_name, task_summary in report.task_summaries.items():
        for policy_name, agg in task_summary.policy_summaries.items():
            row = {
                "policy_name":      f"{policy_name}",
                "task":             task_name,
                "compression_ratio": agg.get("compression_ratio",  {}).get("mean"),
                "perplexity":        agg.get("perplexity",          {}).get("mean"),
                "token_f1":          agg.get("token_f1",            {}).get("mean"),
                "rouge1_f1":         agg.get("rouge1_f1",           {}).get("mean"),
                "faithfulness":      agg.get("faithfulness",        {}).get("mean"),
                "syntax_valid":      agg.get("syntax_valid",        {}).get("mean"),
                "identifier_overlap":agg.get("identifier_overlap",  {}).get("mean"),
            }
            all_rows.append(row)

    # 1. Per-task Pareto frontier (compression ratio vs perplexity)
    for task_name in tasks:
        task_rows = [r for r in all_rows if r["task"] == task_name]
        if task_rows:
            plot_pareto_frontier(
                task_rows,
                quality_metric="perplexity",
                title=f"Pareto Frontier — {task_name.title()} (PPL vs Compression)",
                save_path=f"{plot_dir}/pareto_{task_name}.png",
            )
            plot_compression_vs_quality(
                task_rows,
                title=f"Compression vs Quality — {task_name.title()}",
                save_path=f"{plot_dir}/compression_quality_{task_name}.png",
            )

    # 2. Radar chart per task (multi-metric policy comparison)
    for task_name in tasks:
        task_rows = [r for r in all_rows if r["task"] == task_name]
        if len(task_rows) >= 2:
            metrics = {
                "qa":            ["compression_ratio", "perplexity", "token_f1", "rouge1_f1"],
                "summarization": ["compression_ratio", "perplexity", "rouge1_f1", "faithfulness"],
                "code":          ["compression_ratio", "perplexity", "syntax_valid", "identifier_overlap"],
            }.get(task_name, ["compression_ratio", "perplexity"])
            plot_radar_chart(
                task_rows,
                metrics=metrics,
                title=f"Policy Radar — {task_name.title()}",
                save_path=f"{plot_dir}/radar_{task_name}.png",
            )

    # 3. Cross-task radar (averaged across all tasks per policy)
    if len(tasks) > 1:
        policy_names = list({r["policy_name"] for r in all_rows})
        cross_task_rows = []
        for pol in policy_names:
            pol_rows = [r for r in all_rows if r["policy_name"] == pol]
            if not pol_rows:
                continue
            def _avg(key):
                vals = [r[key] for r in pol_rows if r.get(key) is not None]
                return sum(vals) / len(vals) if vals else None
            cross_task_rows.append({
                "policy_name":       pol,
                "compression_ratio": _avg("compression_ratio"),
                "perplexity":        _avg("perplexity"),
                "token_f1":          _avg("token_f1"),
                "rouge1_f1":         _avg("rouge1_f1"),
            })
        if len(cross_task_rows) >= 2:
            plot_radar_chart(
                cross_task_rows,
                metrics=["compression_ratio", "perplexity", "token_f1", "rouge1_f1"],
                title="Cross-Task Policy Comparison (Averaged)",
                save_path=f"{plot_dir}/radar_cross_task.png",
            )

    print(f"\n[advanced_eval] All plots saved to {plot_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Printing helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_cross_task_summary(report, policies: list[str]) -> None:
    """Print a cross-task summary table: policy vs avg metrics across all tasks."""
    print(f"\n{'═'*85}")
    print(f"  CROSS-TASK SUMMARY  (averaged across all tasks)")
    print(f"{'═'*85}")
    print(f"  {'Policy':<22}  {'AvgRatio':>9}  {'AvgPPL':>9}  {'AvgF1':>9}  {'AvgROUGE-1':>11}")
    print(f"{'─'*85}")

    for pol in policies:
        ratios, ppls, f1s, r1s = [], [], [], []
        for task_summary in report.task_summaries.values():
            agg = task_summary.policy_summaries.get(pol, {})
            def _m(key):
                return agg.get(key, {}).get("mean")
            if _m("compression_ratio") is not None: ratios.append(_m("compression_ratio"))
            if _m("perplexity")        is not None: ppls.append(_m("perplexity"))
            if _m("token_f1")          is not None: f1s.append(_m("token_f1"))
            if _m("rouge1_f1")         is not None: r1s.append(_m("rouge1_f1"))

        avg_ratio = sum(ratios)/len(ratios) if ratios else None
        avg_ppl   = sum(ppls)  /len(ppls)   if ppls   else None
        avg_f1    = sum(f1s)   /len(f1s)    if f1s    else None
        avg_r1    = sum(r1s)   /len(r1s)    if r1s    else None

        ratio_s = f"{avg_ratio:.3f}" if avg_ratio is not None else "  N/A"
        ppl_s   = f"{avg_ppl:.2f}"   if avg_ppl   is not None else "   N/A"
        f1_s    = f"{avg_f1:.3f}"    if avg_f1    is not None else "  N/A"
        r1_s    = f"{avg_r1:.3f}"    if avg_r1    is not None else "   N/A"
        print(f"  {pol:<22}  {ratio_s:>9}  {ppl_s:>9}  {f1_s:>9}  {r1_s:>11}")

    print(f"{'═'*85}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    t_start = time.time()

    # Load config file if provided
    cfg = {}
    if args.config:
        cfg = load_config_file(args.config)
        print(f"[advanced_eval] Loaded config: {args.config}")

    model_name   = cfg.get("model",      args.model)
    dtype        = cfg.get("dtype",      args.dtype)
    device_map   = cfg.get("device_map", args.device_map)
    trust_remote = cfg.get("trust_remote_code", args.trust_remote_code)
    n_samples    = cfg.get("benchmark", {}).get("num_seeds", args.n_samples)
    keep_last    = cfg.get("policies", {}).get("recency_window", {}).get("keep_last", args.keep_last)
    keep_top     = cfg.get("policies", {}).get("attention_retention", {}).get("keep_top", args.keep_top)
    clusters     = cfg.get("policies", {}).get("hybrid_cluster", {}).get("clusters", args.clusters)
    results_dir  = cfg.get("output", {}).get("results_dir", args.results_dir)
    plot_dir     = cfg.get("output", {}).get("plots_dir", args.plot_dir)
    do_plot      = cfg.get("output", {}).get("export_json", False) or args.plot

    tasks    = args.tasks
    pol_keys = args.policies

    # ── Banner ──
    print(f"\n{'═'*70}")
    print(f"  KV-Cache Compression — Advanced Evaluation")
    print(f"  Model   : {model_name}")
    print(f"  Tasks   : {tasks}")
    print(f"  Policies: {pol_keys}")
    print(f"  Samples : {n_samples} per task")
    print(f"{'═'*70}\n")

    # ── Load model ──
    print(f"Loading model: {model_name}  (dtype={dtype})")
    tokenizer, model = load_causal_lm(
        model_name,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote,
    )
    print(f"Model loaded.\n")

    # ── Build policies ──
    all_policies_map = build_all_policies(keep_last=keep_last, keep_top=keep_top, clusters=clusters)
    selected_policies = [all_policies_map[k] for k in pol_keys if k in all_policies_map]
    print(f"Policies to evaluate ({len(selected_policies)}): {[p.name for p in selected_policies]}\n")

    # ── Run task runner ──
    config = TaskConfig(
        tasks=tasks,
        n_samples=n_samples,
        verbose=True,
    )
    runner = TaskRunner(model, tokenizer, config=config)
    report = runner.run_all(selected_policies)

    # ── Cross-task summary ──
    print_cross_task_summary(report, pol_keys)

    # ── Export results ──
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    json_path = f"{results_dir}/advanced_eval_results.json"
    csv_path  = f"{results_dir}/advanced_eval_results.csv"
    runner.export(report, json_path=json_path, csv_path=csv_path)

    # ── Generate plots ──
    if do_plot or args.plot:
        _plot_all(report, plot_dir=plot_dir, tasks=tasks)

    elapsed = time.time() - t_start
    print(f"\n[advanced_eval] Done in {elapsed:.1f}s")
    print(f"[advanced_eval] Results → {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
