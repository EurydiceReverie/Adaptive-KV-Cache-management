"""
Unified Task Runner for KV-Cache Compression Evaluation.

Runs all three evaluation tasks (Long-QA, Summarization, Code Completion)
across all configured compression policies in a single call, aggregating
results into a unified report.

Features
--------
- Run any subset of tasks: --tasks qa sum code
- Per-task and cross-task summaries
- JSON + CSV export
- Optional radar/Pareto plots comparing policies across tasks
- Progress reporting

Usage (Python API)
------------------
    from kv_cache_compression.eval.task_runner import TaskRunner, TaskConfig
    from kv_cache_compression.cache.prune import RecencyWindowPolicy
    from kv_cache_compression.cache.advanced_policies import SinkTokenPolicy

    policies = [RecencyWindowPolicy(512), SinkTokenPolicy(4, 512)]
    config   = TaskConfig(tasks=["qa", "summarization", "code"], n_samples=5)
    runner   = TaskRunner(model, tokenizer, config=config)
    report   = runner.run_all(policies)
    runner.print_report(report)
    runner.export(report, json_path="results/report.json", csv_path="results/report.csv")

Usage (CLI)
-----------
    python main.py --task all --policy hybrid --policy sink --n-samples 5 --export-csv results/all.csv
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from kv_cache_compression.cache.policies import CompressionPolicy
from kv_cache_compression.eval.long_qa_eval import LongQAEvaluator, get_builtin_samples
from kv_cache_compression.eval.summarization_eval import SummarizationEvaluator, get_builtin_summaries
from kv_cache_compression.eval.code_completion_eval import CodeCompletionEvaluator, get_builtin_code_samples


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

VALID_TASKS = {"qa", "summarization", "code"}


@dataclass
class TaskConfig:
    """Configuration for the unified task runner."""
    tasks: list[str] = field(default_factory=lambda: ["qa", "summarization", "code"])
    n_samples: int = 5
    pad_qa_to_length: int = 0    # optional padding for QA context
    verbose: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.tasks) - VALID_TASKS
        if unknown:
            raise ValueError(f"Unknown tasks: {unknown}. Valid: {VALID_TASKS}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-task aggregate summary
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskSummary:
    """Aggregated results for one task across all policies."""
    task: str
    policy_summaries: dict[str, dict]   # policy_name → aggregate dict

    def best_policy(self, metric: str = "compression_ratio") -> str | None:
        """Return the policy name with the best mean value for a metric."""
        best_name, best_val = None, None
        lower_is_better = {"compression_ratio", "perplexity", "continuation_nll"}
        for name, agg in self.policy_summaries.items():
            val = agg.get(metric, {}).get("mean")
            if val is None:
                continue
            if best_val is None:
                best_name, best_val = name, val
            elif metric in lower_is_better and val < best_val:
                best_name, best_val = name, val
            elif metric not in lower_is_better and val > best_val:
                best_name, best_val = name, val
        return best_name


@dataclass
class TaskReport:
    """Complete report from running all tasks across all policies."""
    task_summaries: dict[str, TaskSummary]   # task → TaskSummary
    policies_run: list[str]
    elapsed_seconds: float
    config: TaskConfig

    def to_dict(self) -> dict:
        return {
            "policies": self.policies_run,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "tasks": {
                task: {
                    name: agg
                    for name, agg in ts.policy_summaries.items()
                }
                for task, ts in self.task_summaries.items()
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# Unified Task Runner
# ══════════════════════════════════════════════════════════════════════════════

class TaskRunner:
    """
    Runs Long-QA, Summarization, and Code Completion evaluations across
    multiple compression policies, returning a unified TaskReport.

    Parameters
    ----------
    model      : HuggingFace causal LM
    tokenizer  : matching tokenizer
    config     : TaskConfig instance (defaults to all tasks, 5 samples each)
    """

    def __init__(self, model, tokenizer, config: TaskConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TaskConfig()
        self._qa_eval    = LongQAEvaluator(model, tokenizer, verbose=self.config.verbose)
        self._sum_eval   = SummarizationEvaluator(model, tokenizer, verbose=self.config.verbose)
        self._code_eval  = CodeCompletionEvaluator(model, tokenizer, verbose=self.config.verbose)

    def run_all(self, policies: list[CompressionPolicy]) -> TaskReport:
        """
        Run all configured tasks across all policies.

        Returns a TaskReport with per-task, per-policy aggregated metrics.
        """
        t0 = time.time()
        task_summaries: dict[str, TaskSummary] = {}
        cfg = self.config

        if cfg.verbose:
            print(f"\n{'═'*70}")
            print(f"  Unified Task Runner")
            print(f"  Tasks: {cfg.tasks}  |  Policies: {[p.name for p in policies]}")
            print(f"  Samples per task: {cfg.n_samples}")
            print(f"{'═'*70}")

        # ── Long-form QA ──
        if "qa" in cfg.tasks:
            if cfg.verbose:
                print(f"\n{'─'*70}")
                print(f"  Task: Long-Form QA")
                print(f"{'─'*70}")
            samples = get_builtin_samples(n=cfg.n_samples, pad_to_length=cfg.pad_qa_to_length)
            results = self._qa_eval.run_all(policies, samples=samples)
            task_summaries["qa"] = TaskSummary(
                task="qa",
                policy_summaries={
                    name: self._qa_eval.aggregate(res)
                    for name, res in results.items()
                },
            )

        # ── Summarization ──
        if "summarization" in cfg.tasks:
            if cfg.verbose:
                print(f"\n{'─'*70}")
                print(f"  Task: Summarization")
                print(f"{'─'*70}")
            samples = get_builtin_summaries(n=cfg.n_samples)
            results = self._sum_eval.run_all(policies, samples=samples)
            task_summaries["summarization"] = TaskSummary(
                task="summarization",
                policy_summaries={
                    name: self._sum_eval.aggregate(res)
                    for name, res in results.items()
                },
            )

        # ── Code Completion ──
        if "code" in cfg.tasks:
            if cfg.verbose:
                print(f"\n{'─'*70}")
                print(f"  Task: Code Completion")
                print(f"{'─'*70}")
            samples = get_builtin_code_samples(n=cfg.n_samples)
            results = self._code_eval.run_all(policies, samples=samples)
            task_summaries["code"] = TaskSummary(
                task="code",
                policy_summaries={
                    name: self._code_eval.aggregate(res)
                    for name, res in results.items()
                },
            )

        elapsed = time.time() - t0
        report = TaskReport(
            task_summaries=task_summaries,
            policies_run=[p.name for p in policies],
            elapsed_seconds=elapsed,
            config=cfg,
        )
        if cfg.verbose:
            self.print_report(report)
        return report

    def print_report(self, report: TaskReport) -> None:
        """Print a formatted summary of the full report."""
        print(f"\n{'═'*80}")
        print(f"  UNIFIED EVALUATION REPORT  ({report.elapsed_seconds:.1f}s)")
        print(f"  Policies: {', '.join(report.policies_run)}")
        print(f"{'═'*80}")

        for task, ts in report.task_summaries.items():
            task_label = {"qa": "Long-Form QA", "summarization": "Summarization",
                          "code": "Code Completion"}.get(task, task)
            print(f"\n  ┌─ {task_label}")

            # Choose representative metrics per task
            metric_keys = {
                "qa":            ["compression_ratio", "perplexity", "token_f1", "rouge1_f1"],
                "summarization": ["compression_ratio", "perplexity", "rouge1_f1", "faithfulness"],
                "code":          ["compression_ratio", "perplexity", "syntax_valid", "identifier_overlap"],
            }.get(task, ["compression_ratio", "perplexity"])

            header = f"  │  {'Policy':<22}" + "".join(f"  {m[:10]:>10}" for m in metric_keys)
            print(header)
            print(f"  │  {'-'*22}" + "──────────" * len(metric_keys))

            for policy_name in report.policies_run:
                agg = ts.policy_summaries.get(policy_name, {})
                row = f"  │  {policy_name:<22}"
                for m in metric_keys:
                    val = agg.get(m, {}).get("mean")
                    row += f"  {val:>10.3f}" if val is not None else f"  {'N/A':>10}"
                print(row)

            best = ts.best_policy("perplexity")
            if best:
                print(f"  └─ Best (lowest PPL): {best}")

        print(f"\n{'═'*80}")

    def export(
        self,
        report: TaskReport,
        json_path: str | None = None,
        csv_path: str | None = None,
    ) -> None:
        """Export the report to JSON and/or CSV."""
        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            print(f"[task_runner] JSON saved → {json_path}")

        if csv_path:
            rows = []
            for task, ts in report.task_summaries.items():
                for policy_name, agg in ts.policy_summaries.items():
                    row = {"task": task, "policy": policy_name}
                    for metric, stats in agg.items():
                        if isinstance(stats, dict):
                            row[f"{metric}_mean"] = stats.get("mean")
                            row[f"{metric}_std"]  = stats.get("std")
                        else:
                            row[metric] = stats
                    rows.append(row)

            if rows:
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                all_keys = sorted({k for r in rows for k in r})
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[task_runner] CSV saved → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: run from a config dict (for CLI integration)
# ══════════════════════════════════════════════════════════════════════════════

def run_from_config(
    model,
    tokenizer,
    policies: list[CompressionPolicy],
    *,
    tasks: list[str] | None = None,
    n_samples: int = 5,
    verbose: bool = True,
    json_path: str | None = None,
    csv_path: str | None = None,
) -> TaskReport:
    """
    Convenience wrapper: create TaskRunner, run, export, return report.

    Parameters
    ----------
    model, tokenizer : loaded HuggingFace model + tokenizer
    policies         : list of CompressionPolicy instances to compare
    tasks            : list subset of ["qa", "summarization", "code"] (None = all)
    n_samples        : samples per task
    verbose          : print progress
    json_path        : if set, save JSON report here
    csv_path         : if set, save CSV report here
    """
    config = TaskConfig(
        tasks=tasks or ["qa", "summarization", "code"],
        n_samples=n_samples,
        verbose=verbose,
    )
    runner = TaskRunner(model, tokenizer, config=config)
    report = runner.run_all(policies)
    runner.export(report, json_path=json_path, csv_path=csv_path)
    return report
