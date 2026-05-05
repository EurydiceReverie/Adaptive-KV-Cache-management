"""
Plotting utilities for KV-cache compression experiment results.

Generates comparison charts:
  - Compression ratio vs NLL (quality)
  - Memory saved vs perplexity
  - Latency breakdown (prompt vs compression)
  - Per-policy token reduction bar chart
  - Attention heatmap (token position vs attention score)
  - Pareto frontier (compression vs quality trade-off)
  - Radar / spider chart (multi-metric policy comparison)
  - Budget schedule curves (how budget evolves with context length)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# Consistent colour palette for all charts
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend safe for scripts
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_attention_heatmap(
    token_labels: list[str],
    attention_scores: list[float],
    kept_mask: list[bool] | None = None,
    title: str = "",
    save_path: str | None = None,
    max_tokens: int = 200,
) -> None:
    """
    Attention score heatmap: horizontal bar chart showing per-token importance.

    Parameters
    ----------
    token_labels    : list of token strings (x-axis labels)
    attention_scores: list of per-token attention scores
    kept_mask       : optional bool list — True = token kept, False = evicted
    max_tokens      : cap number of tokens shown (take top-scoring if over limit)
    save_path       : if set, save to file instead of showing
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping heatmap.")
        return

    import matplotlib.pyplot as mpl_plt
    import numpy as np

    n = len(attention_scores)
    if n > max_tokens:
        top_idx = sorted(range(n), key=lambda i: attention_scores[i], reverse=True)[:max_tokens]
        top_idx = sorted(top_idx)
        attention_scores = [attention_scores[i] for i in top_idx]
        token_labels     = [token_labels[i]     for i in top_idx]
        kept_mask        = [kept_mask[i]         for i in top_idx] if kept_mask else None

    n = len(attention_scores)
    fig_h = max(4, n * 0.18)
    fig, ax = mpl_plt.subplots(figsize=(10, fig_h))

    colors = []
    for i in range(n):
        if kept_mask is None:
            colors.append("#4C72B0")
        elif kept_mask[i]:
            colors.append("#55A868")   # green = kept
        else:
            colors.append("#C44E52")   # red = evicted

    y_pos = np.arange(n)
    ax.barh(y_pos, attention_scores, color=colors, height=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:20] for t in token_labels], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Attention Score")
    ax.set_title(title or "Per-Token Attention Scores")

    if kept_mask is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#55A868", label="Kept"),
            Patch(facecolor="#C44E52", label="Evicted"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    mpl_plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved heatmap → {save_path}")
    else:
        mpl_plt.show()
    mpl_plt.close(fig)


def plot_pareto_frontier(
    results: list[dict],
    quality_metric: str = "perplexity",
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Pareto frontier chart: compression ratio vs quality metric.

    Highlights Pareto-optimal policies (not dominated by any other).

    Parameters
    ----------
    results        : list of result dicts with "policy_name", "compression_ratio",
                     and `quality_metric` keys
    quality_metric : metric name for the Y-axis (lower = better, e.g. perplexity)
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping Pareto chart.")
        return

    import matplotlib.pyplot as mpl_plt

    fig, ax = mpl_plt.subplots(figsize=(8, 6))

    policies  = [r["policy_name"]       for r in results]
    ratios    = [r.get("compression_ratio", 1.0) for r in results]
    qualities = [r.get(quality_metric)           for r in results]

    # Compute Pareto frontier
    pareto_mask = []
    for i, (ri, qi) in enumerate(zip(ratios, qualities)):
        if qi is None:
            pareto_mask.append(False)
            continue
        dominated = any(
            rj <= ri and qj is not None and qj <= qi
            for j, (rj, qj) in enumerate(zip(ratios, qualities))
            if j != i
        )
        pareto_mask.append(not dominated)

    for i, (policy, ratio, quality) in enumerate(zip(policies, ratios, qualities)):
        if quality is None:
            continue
        color = _PALETTE[i % len(_PALETTE)]
        marker = "*" if pareto_mask[i] else "o"
        size   = 200 if pareto_mask[i] else 80
        ax.scatter(ratio, quality, s=size, color=color, marker=marker,
                   zorder=5, label=policy + (" ★" if pareto_mask[i] else ""))
        ax.annotate(policy, (ratio, quality), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=color)

    # Draw Pareto frontier line
    pareto_pts = sorted(
        [(ratio, q) for ratio, q, m in zip(ratios, qualities, pareto_mask) if m and q is not None],
        key=lambda x: x[0],
    )
    if len(pareto_pts) >= 2:
        px, py = zip(*pareto_pts)
        ax.step(px, py, where="post", linestyle="--", color="gray", alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("Compression Ratio  (lower = more compressed)")
    ax.set_ylabel(f"{quality_metric}  (lower = better)")
    ax.set_title(title or f"Pareto Frontier: Compression vs {quality_metric.title()}")
    ax.legend(fontsize=7, loc="upper left")
    mpl_plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved Pareto chart → {save_path}")
    else:
        mpl_plt.show()
    mpl_plt.close(fig)


def plot_radar_chart(
    results: list[dict],
    metrics: list[str] | None = None,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Radar / spider chart for multi-metric policy comparison.

    Each axis represents one metric (normalised 0→1, higher = better).
    Each policy is drawn as a filled polygon.

    Parameters
    ----------
    results : list of result dicts (one per policy)
    metrics : metrics to include on axes (auto-selected if None)
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping radar chart.")
        return

    import matplotlib.pyplot as mpl_plt
    import numpy as np

    if metrics is None:
        metrics = ["compression_ratio", "tokens_saved_pct", "prompt_seconds"]
        # Add quality metrics if present
        for m in ["perplexity", "continuation_nll", "token_f1_score", "rouge1_f1"]:
            if any(m in r for r in results):
                metrics.append(m)
        metrics = metrics[:8]   # cap at 8 axes

    # Normalise metrics to 0–1 (handle "lower is better" inversion)
    lower_is_better = {"perplexity", "continuation_nll", "prompt_seconds",
                       "compression_seconds", "compression_ratio"}

    norm_data: dict[str, list[float]] = {}
    for metric in metrics:
        vals = [float(r.get(metric) or 0.0) for r in results]
        vmin, vmax = min(vals), max(vals)
        spread = vmax - vmin or 1.0
        normalised = [(v - vmin) / spread for v in vals]
        # Invert so higher = better on radar
        if metric in lower_is_better:
            normalised = [1.0 - v for v in normalised]
        norm_data[metric] = normalised

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = mpl_plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for i, result in enumerate(results):
        values = [norm_data[m][i] for m in metrics]
        values += values[:1]
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2, label=result["policy_name"])
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6)
    ax.set_title(title or "Multi-Metric Policy Comparison", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    mpl_plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved radar chart → {save_path}")
    else:
        mpl_plt.show()
    mpl_plt.close(fig)


def plot_budget_schedule(
    scheduler,
    seq_lens: list[int] | None = None,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot how a BudgetScheduler varies its compression budget over context lengths.

    Parameters
    ----------
    scheduler : any BudgetScheduler instance
    seq_lens  : list of sequence lengths to evaluate (auto-generated if None)
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping budget schedule chart.")
        return

    import matplotlib.pyplot as mpl_plt

    if seq_lens is None:
        seq_lens = list(range(128, 8193, 128))

    budgets = [scheduler.get_budget(l) for l in seq_lens]
    ratios  = [b.compression_ratio for b in budgets]
    totals  = [b.total_budget       for b in budgets]

    fig, (ax1, ax2) = mpl_plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(seq_lens, ratios, color="#4C72B0", linewidth=2)
    ax1.fill_between(seq_lens, ratios, alpha=0.15, color="#4C72B0")
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("Compression Ratio  (fraction of tokens kept)")
    ax1.set_title("Adaptive Compression Ratio vs Context Length")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1.0, linestyle="--", color="gray", alpha=0.5, label="No compression")
    ax1.legend(fontsize=8)

    ax2.plot(seq_lens, totals, color="#DD8452", linewidth=2)
    ax2.fill_between(seq_lens, totals, alpha=0.15, color="#DD8452")
    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("Absolute Budget (total tokens kept)")
    ax2.set_title("Absolute Budget vs Context Length")
    ax2.plot(seq_lens, seq_lens, linestyle="--", color="gray", alpha=0.5, label="No compression (y=x)")
    ax2.legend(fontsize=8)

    sched_name = getattr(scheduler, "__class__", type(scheduler)).__name__
    fig.suptitle(title or f"Budget Schedule: {sched_name}", fontsize=13, fontweight="bold")
    mpl_plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved budget schedule chart → {save_path}")
    else:
        mpl_plt.show()
    mpl_plt.close(fig)


def plot_streaming_history(
    history: list,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Plot the compression history from a StreamingCompressor.

    Shows tokens-before and tokens-after at each compression step.

    Parameters
    ----------
    history : list of StreamingStats objects from compressor.history
    """
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping streaming history chart.")
        return

    import matplotlib.pyplot as mpl_plt

    steps       = [s.step              for s in history]
    lens_before = [s.seq_len_before    for s in history]
    lens_after  = [s.seq_len_after     for s in history]
    compressed  = [s.compressed        for s in history]

    fig, ax = mpl_plt.subplots(figsize=(10, 5))
    ax.plot(steps, lens_before, color="#4C72B0", linewidth=1.5, label="Tokens before compression")
    ax.plot(steps, lens_after,  color="#55A868", linewidth=1.5, label="Tokens after compression")

    compress_steps = [s.step for s in history if s.compressed]
    compress_lens  = [s.seq_len_after for s in history if s.compressed]
    ax.scatter(compress_steps, compress_lens, color="#C44E52", s=50, zorder=5, label="Compression event")

    ax.set_xlabel("Generation Step")
    ax.set_ylabel("KV Cache Size (tokens)")
    ax.set_title(title or "Streaming KV-Cache Size Over Generation")
    ax.legend(fontsize=9)
    mpl_plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved streaming history chart → {save_path}")
    else:
        mpl_plt.show()
    mpl_plt.close(fig)


def plot_compression_vs_quality(results: list[dict], title: str = "", save_path: str | None = None) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping chart.")
        return

    import matplotlib.pyplot as mpl_plt

    policies   = [r["policy_name"] for r in results]
    comp_ratio = [r.get("compression_ratio", 1.0) for r in results]
    nll_vals   = [r.get("continuation_nll") for r in results]
    ppl_vals   = [math.exp(n) if n is not None else None for n in nll_vals]

    fig, axes = mpl_plt.subplots(1, 2, figsize=(12, 5))

    # Left: compression ratio per policy
    ax = axes[0]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(policies, comp_ratio, color=colors[: len(policies)])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Compression Ratio  (lower = more compressed)")
    ax.set_title("Token Compression Ratio by Policy")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, comp_ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Right: perplexity per policy
    ax2 = axes[1]
    valid_ppls = [p if p is not None else 0.0 for p in ppl_vals]
    bars2 = ax2.bar(policies, valid_ppls, color=colors[: len(policies)])
    ax2.set_ylabel("Perplexity  (lower = better quality)")
    ax2.set_title("Generation Quality (Perplexity) by Policy")
    ax2.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars2, valid_ppls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    mpl_plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved chart → {save_path}")
    else:
        mpl_plt.show()

    mpl_plt.close(fig)


def plot_memory_vs_quality(results: list[dict], title: str = "", save_path: str | None = None) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping chart.")
        return

    import matplotlib.pyplot as mpl_plt

    policies    = [r["policy_name"] for r in results]
    saved_pct   = [r.get("tokens_saved_pct", 0.0) for r in results]
    nll_vals    = [r.get("continuation_nll") for r in results]
    ppl_vals    = [math.exp(n) if n is not None else None for n in nll_vals]

    fig, ax = mpl_plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (policy, saved, ppl) in enumerate(zip(policies, saved_pct, ppl_vals)):
        if ppl is not None:
            ax.scatter(saved, ppl, s=120, color=colors[i % len(colors)], label=policy, zorder=5)
            ax.annotate(policy, (saved, ppl), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Tokens Saved (%)")
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title(title or "Memory Saved vs Generation Quality")
    ax.legend(fontsize=8)
    mpl_plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved chart → {save_path}")
    else:
        mpl_plt.show()

    mpl_plt.close(fig)


def plot_latency_breakdown(results: list[dict], title: str = "", save_path: str | None = None) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        print("[plotting] matplotlib not available — skipping chart.")
        return

    import matplotlib.pyplot as mpl_plt
    import numpy as np

    policies       = [r["policy_name"] for r in results]
    prompt_times   = [r.get("prompt_seconds", 0.0) for r in results]
    compress_times = [r.get("compression_seconds", 0.0) for r in results]

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = mpl_plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, prompt_times, width, label="Prompt forward pass", color="#4C72B0")
    ax.bar(x + width / 2, compress_times, width, label="Compression step", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15)
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title or "Latency Breakdown by Policy")
    ax.legend()
    mpl_plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plotting] Saved chart → {save_path}")
    else:
        mpl_plt.show()

    mpl_plt.close(fig)


def generate_all_plots(results_json_path: str, output_dir: str = "experiments/results/plots") -> None:
    """
    Load a results JSON (produced by compare_policies.py --output-json) and
    generate all standard charts into output_dir.
    """
    with open(results_json_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for task_name, task_results in data.items():
        plot_compression_vs_quality(
            task_results,
            title=f"Compression vs Quality — {task_name}",
            save_path=str(out / f"{task_name}_compression_quality.png"),
        )
        plot_memory_vs_quality(
            task_results,
            title=f"Memory Saved vs Quality — {task_name}",
            save_path=str(out / f"{task_name}_memory_quality.png"),
        )
        plot_latency_breakdown(
            task_results,
            title=f"Latency Breakdown — {task_name}",
            save_path=str(out / f"{task_name}_latency.png"),
        )

    print(f"\n[plotting] All charts saved to: {output_dir}")
