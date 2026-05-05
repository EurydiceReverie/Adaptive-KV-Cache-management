"""
Phase 5 vs ALL Policies — Comprehensive Comparison Experiment
=============================================================
Runs every compression policy (Phases 1-5) on a rich set of prompts and
produces a detailed JSON + printed table showing:
  - compression ratio, memory saved
  - NLL (quality loss), perplexity
  - which tokens were kept / forgotten / why

Usage:
    # Quick run (CPU, TinyLlama)
    python experiments/phase5_compare_all.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --n-prompts 3 --output-dir experiments/results/phase5

    # Full run with local model
    python experiments/phase5_compare_all.py \
        --model models/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
        --n-prompts 10 --output-dir experiments/results/phase5

Results saved to:
    experiments/results/phase5/comparison_results.json
    experiments/results/phase5/summary_table.txt
"""

from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.cache.kv_cache import KVCacheInspector, to_legacy_tuple
from kv_cache_compression.cache.prune import aggregate_attention_scores, build_attention_indices
from kv_cache_compression.cache.policies import PolicyContext
from kv_cache_compression.cache.cluster import HybridClusterPolicy
from kv_cache_compression.cache.layerwise import LayerwiseHybridPolicy
from kv_cache_compression.cache.prune import RecencyWindowPolicy, AttentionRetentionPolicy
from kv_cache_compression.cache.advanced_policies import (
    SinkTokenPolicy, HeavyHitterPolicy, QuantizedCachePolicy, ScissorHandsPolicy
)
from kv_cache_compression.cache.learned_memory import LearnedMemoryPolicy
from kv_cache_compression.eval.benchmark import BenchmarkRunner, BenchmarkSample, _key_norm_scores

# ── Rich prompt library covering diverse long-context scenarios ───────────────
ALL_PROMPTS = [
    BenchmarkSample(
        prompt=(
            "The transformer architecture, introduced by Vaswani et al. in 2017, "
            "revolutionised natural language processing. At its core, the self-attention "
            "mechanism allows every token to attend to every other token in the sequence. "
            "However, this comes at a quadratic cost: for a sequence of N tokens, the "
            "attention matrix is N×N. For N=4096, that is 16 million entries per layer. "
            "With 32 layers and float16 precision, just the attention weights consume "
            "over 1 GB of GPU memory — before the model's parameters even count. "
            "KV-cache compression directly attacks this bottleneck by reducing the "
            "effective sequence length the model must attend over, without retraining. "
            "The key insight is that not all tokens are equally important at every step."
        ),
        continuation="The compression ratio achieved by these methods depends on the budget.",
        answer="compression"
    ),
    BenchmarkSample(
        prompt=(
            "In 1969, Neil Armstrong became the first human to walk on the Moon. "
            "The Apollo 11 mission launched on July 16 and landed on July 20. "
            "Armstrong's famous words were: 'That's one small step for man, one giant leap "
            "for mankind.' The mission carried three astronauts: Armstrong, Buzz Aldrin, "
            "and Michael Collins. Collins remained in orbit while the others descended. "
            "The lunar module was named Eagle. The command module was named Columbia. "
            "They collected 47.5 pounds of lunar material. The mission lasted 8 days. "
            "It was broadcast live to 600 million viewers worldwide — the largest TV "
            "audience in history at that time. NASA's budget in 1966 was $4.5 billion, "
            "which is roughly $34 billion in 2023 dollars. The Saturn V rocket used "
            "consumed 20 tonnes of fuel per second during liftoff. The computer guidance "
            "system had only 4KB of RAM — less than a modern calculator. Despite this, "
            "the mission succeeded due to exceptional engineering and human problem-solving."
        ),
        continuation="The first Moon landing was a triumph of human engineering.",
        answer="Moon landing"
    ),
    BenchmarkSample(
        prompt=(
            "Climate change is driven primarily by the accumulation of greenhouse gases "
            "such as CO2, methane, and nitrous oxide in the atmosphere. Since the "
            "Industrial Revolution, atmospheric CO2 has risen from 280 ppm to over 420 ppm. "
            "The IPCC's Sixth Assessment Report (2021) concluded that human influence has "
            "warmed the climate at an unprecedented rate. Global surface temperature has "
            "increased by approximately 1.1°C above pre-industrial levels. At 1.5°C of "
            "warming, 70-90% of coral reefs are projected to decline. At 2°C, virtually "
            "all coral reefs would be lost. Arctic sea ice has declined by 13% per decade "
            "since 1979. Sea levels have risen 20 cm since 1900 and are accelerating. "
            "Extreme weather events — hurricanes, droughts, floods — are becoming more "
            "frequent and intense. Renewable energy (solar, wind) has seen cost reductions "
            "of over 90% in the last decade, making clean energy economically competitive. "
            "However, global emissions are still rising, reaching 36.8 Gt CO2 in 2023."
        ),
        continuation="Reducing emissions requires urgent policy action globally.",
        answer="climate"
    ),
    BenchmarkSample(
        prompt=(
            "Python is a high-level, interpreted programming language created by Guido "
            "van Rossum and released in 1991. It emphasises code readability and simplicity. "
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
            "This recursive implementation, while elegant, has exponential time complexity O(2^n). "
            "A better approach uses dynamic programming:\n"
            "def fibonacci_dp(n):\n    if n <= 1:\n        return n\n"
            "    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b\n"
            "This runs in O(n) time and O(1) space. Python's popularity stems from its vast "
            "ecosystem: NumPy for arrays, Pandas for data, PyTorch and TensorFlow for deep "
            "learning, FastAPI for web services, and thousands more packages on PyPI. "
            "Python consistently ranks #1 in the TIOBE Index and Stack Overflow surveys."
        ),
        continuation="Python's simplicity makes it ideal for rapid prototyping.",
        answer="Python"
    ),
    BenchmarkSample(
        prompt=(
            "The human brain contains approximately 86 billion neurons, each connected "
            "to roughly 7,000 other neurons via synapses, giving around 100 trillion "
            "synaptic connections total. Neural signals travel at speeds from 0.5 to "
            "120 metres per second, depending on myelination. The brain consumes about "
            "20 watts of power — roughly the same as a dim light bulb — yet performs "
            "computations that no supercomputer can replicate. Memory consolidation occurs "
            "during sleep, particularly in deep slow-wave sleep and REM stages. "
            "The hippocampus is central to forming new memories, while the amygdala "
            "processes emotional responses. Neuroplasticity — the brain's ability to "
            "reorganise itself — remains active throughout life, though it peaks in "
            "childhood. Language is processed primarily in Broca's area (production) "
            "and Wernicke's area (comprehension) in the left hemisphere for most people. "
            "Modern fMRI techniques can detect blood-oxygen changes with millimetre "
            "spatial resolution, enabling researchers to map cognitive functions."
        ),
        continuation="Neuroscience continues to reveal the mechanisms of consciousness.",
        answer="brain"
    ),
    BenchmarkSample(
        prompt=(
            "Large language models (LLMs) such as GPT-4, Claude, and Gemini are trained "
            "on trillions of tokens of text using the transformer architecture. Training "
            "GPT-4 reportedly cost over $100 million in compute alone. These models learn "
            "by predicting the next token in a sequence, a task called autoregressive "
            "language modelling. Despite this simple objective, emergent capabilities "
            "arise at scale: chain-of-thought reasoning, few-shot learning, code generation, "
            "and mathematical problem-solving. The context window — how many tokens the "
            "model can process at once — has grown from 2K tokens (GPT-2) to 1M+ tokens "
            "(Gemini 1.5 Pro). However, longer contexts mean larger KV caches. "
            "At 1M tokens with a 32-layer model and head_dim=128, the KV cache alone "
            "requires 128 GB of memory in float16. This is why KV-cache compression "
            "is one of the most active research areas in LLM inference optimisation."
        ),
        continuation="Efficient inference is critical for deploying LLMs at scale.",
        answer="LLM inference"
    ),
    BenchmarkSample(
        prompt=(
            "The history of mathematics spans millennia. Ancient Egyptians used fractions "
            "around 1650 BCE (Rhind Papyrus). Euclid formalised geometry (~300 BCE). "
            "Archimedes approximated pi using inscribed polygons. Al-Khwarizmi invented "
            "algebra in the 9th century — the word 'algorithm' comes from his name. "
            "Newton and Leibniz independently invented calculus in the 17th century. "
            "Euler introduced e, i, and the famous identity e^(iπ)+1=0. Gauss proved "
            "the fundamental theorem of algebra. Riemann proposed his famous hypothesis "
            "in 1859, still unproven. Gödel's incompleteness theorems (1931) showed that "
            "any consistent formal system contains true statements that cannot be proved. "
            "Turing formalised computation in 1936, laying the foundation for computer science. "
            "The four-colour theorem was proved in 1976 — the first major proof using computers. "
            "Today, mathematics underpins physics, engineering, cryptography, AI, and finance."
        ),
        continuation="Mathematics is the language in which the laws of nature are written.",
        answer="mathematics"
    ),
    BenchmarkSample(
        prompt=(
            "Quantum computing harnesses quantum mechanical phenomena — superposition, "
            "entanglement, and interference — to perform computations. A classical bit "
            "is either 0 or 1. A qubit can exist in a superposition of both simultaneously. "
            "n qubits can represent 2^n states simultaneously. Shor's algorithm can factor "
            "large numbers in polynomial time, threatening RSA encryption. Grover's algorithm "
            "speeds up search from O(N) to O(√N). Current quantum computers (IBM, Google, IonQ) "
            "have 50-1000+ qubits but suffer from high error rates — a problem called decoherence. "
            "Quantum error correction requires ~1000 physical qubits per logical qubit. "
            "Google claimed 'quantum supremacy' in 2019 for a specific sampling task. "
            "Practical quantum advantage for real-world problems remains years away. "
            "However, quantum simulation — modelling molecules for drug discovery — "
            "is expected to be one of the first commercially useful quantum applications."
        ),
        continuation="Quantum computers promise exponential speedups for specific problems.",
        answer="quantum"
    ),
    BenchmarkSample(
        prompt=(
            "The Amazon rainforest covers 5.5 million square kilometres across 9 countries, "
            "primarily Brazil. It is home to 10% of all species on Earth, including "
            "40,000 plant species, 1,300 bird species, 3,000 freshwater fish, and "
            "over 2.5 million insect species. The forest produces 20% of the world's "
            "oxygen and absorbs billions of tonnes of CO2 annually. "
            "Deforestation has removed 17% of the original forest since 1970. "
            "The Amazon River discharges 20% of all fresh water entering the world's oceans. "
            "Indigenous peoples have lived in the Amazon for over 11,000 years. "
            "Today, approximately 400 distinct indigenous groups call it home. "
            "Researchers fear a 'tipping point' at 20-25% deforestation, beyond which "
            "the forest could irreversibly transition to savannah, releasing stored carbon "
            "and disrupting rainfall patterns across South America."
        ),
        continuation="Protecting the Amazon is critical for global climate stability.",
        answer="Amazon"
    ),
    BenchmarkSample(
        prompt=(
            "Attention is all you need — or is it? The original transformer used multi-head "
            "self-attention with queries Q, keys K, and values V computed as linear projections "
            "of the input embeddings. The attention output is: softmax(QK^T/√d_k)V. "
            "During autoregressive generation, past K and V vectors must be stored — "
            "this is the KV cache. For each new token generated, the model appends new "
            "K and V vectors to the cache. After generating 2048 tokens with a 32-layer, "
            "32-head model with head_dim=128: cache size = 2 * 32 * 32 * 2048 * 128 * 2 bytes "
            "= 1,073,741,824 bytes = 1 GB. Each additional token adds 32 * 32 * 2 * 128 * 2 "
            "= 524,288 bytes = 0.5 MB. At 8K tokens: 4 GB. At 32K tokens: 16 GB. "
            "This linear growth is why long-context generation is so memory-intensive, "
            "and why compression techniques that reduce effective sequence length are "
            "essential for practical deployment of long-context LLMs."
        ),
        continuation="The KV cache grows linearly with sequence length and must be compressed.",
        answer="KV cache"
    ),
]


def get_all_policies(args) -> list:
    """Instantiate all 9 policies with the given budget args."""
    budget = args.keep_last
    top = args.keep_top
    return [
        ("recency",    RecencyWindowPolicy(keep_last_tokens=budget)),
        ("attention",  AttentionRetentionPolicy(keep_last_tokens=budget, keep_top_tokens=top)),
        ("hybrid",     HybridClusterPolicy(keep_last_tokens=budget, keep_top_tokens=top, cluster_tokens=args.clusters)),
        ("layerwise",  LayerwiseHybridPolicy(keep_last_tokens=budget, keep_top_tokens=top)),
        ("sink",       SinkTokenPolicy(num_sink_tokens=4, keep_last_tokens=budget)),
        ("h2o",        HeavyHitterPolicy(budget=budget, num_sink_tokens=4)),
        ("quant",      QuantizedCachePolicy()),
        ("scissor",    ScissorHandsPolicy(num_sink_tokens=4, keep_top_tokens=top, keep_last_tokens=budget // 2)),
        ("learned",    LearnedMemoryPolicy(n_memory=args.n_memory, optim_steps=args.optim_steps,
                                           lr=args.memory_lr, always_keep_last=args.keep_last_learned)),
    ]


def run_comparison(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "═"*65)
    print("  KV-Cache Compression — Phase 5 vs All Policies Comparison")
    print("═"*65)
    print(f"  Model      : {args.model}")
    print(f"  Prompts    : {args.n_prompts}/{len(ALL_PROMPTS)}")
    print(f"  Budget     : keep_last={args.keep_last}, keep_top={args.keep_top}")
    print(f"  Learned    : n_memory={args.n_memory}, steps={args.optim_steps}")
    print(f"  Output dir : {args.output_dir}")
    print("═"*65 + "\n")

    print("⏳ Loading model...")
    t0 = time.time()
    tokenizer, model = load_causal_lm(
        args.model, device_map=args.device_map, dtype=args.dtype
    )
    device = next(model.parameters()).device
    print(f"✅ Model loaded in {time.time()-t0:.1f}s on {device}\n")

    runner = BenchmarkRunner(model, tokenizer)
    prompts = ALL_PROMPTS[:args.n_prompts]
    policies = get_all_policies(args)

    all_results = []

    for pi, (pol_name, policy) in enumerate(policies):
        print(f"\n{'─'*65}")
        print(f"  [{pi+1}/{len(policies)}] Policy: {pol_name.upper()}")
        print(f"{'─'*65}")

        policy_results = []
        for si, sample in enumerate(prompts):
            print(f"    Prompt {si+1}/{len(prompts)}: {sample.prompt[:60]}...")
            try:
                t_start = time.time()
                summary = runner.run(sample, policy)
                elapsed = time.time() - t_start

                ratio = summary.compressed_tokens / max(summary.original_tokens, 1)
                saved_mb = (summary.original_bytes - summary.compressed_bytes) / (1024**2)
                nll = summary.continuation_nll or 0.0
                ppl = round(2**nll, 4) if nll else None

                result = {
                    "policy": pol_name,
                    "prompt_preview": sample.prompt[:80],
                    "answer_topic": sample.answer,
                    "original_tokens": summary.original_tokens,
                    "compressed_tokens": summary.compressed_tokens,
                    "compression_ratio": round(ratio, 4),
                    "memory_saved_mb": round(saved_mb, 3),
                    "original_mb": round(summary.original_bytes / 1024**2, 3),
                    "compressed_mb": round(summary.compressed_bytes / 1024**2, 3),
                    "continuation_nll": round(nll, 5),
                    "perplexity": ppl,
                    "total_seconds": round(elapsed, 2),
                    "metadata": summary.metadata,
                }
                policy_results.append(result)

                print(f"      tokens: {summary.original_tokens} → {summary.compressed_tokens} "
                      f"({ratio*100:.1f}%)  saved {saved_mb:.2f} MB  "
                      f"NLL={nll:.4f}  PPL={ppl}  [{elapsed:.1f}s]")

            except Exception as e:
                print(f"      ⚠️  ERROR: {e}")
                policy_results.append({"policy": pol_name, "error": str(e),
                                       "prompt_preview": sample.prompt[:80]})

        all_results.extend(policy_results)

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Raw results saved to: {json_path}")

    # ── Print summary table ────────────────────────────────────────────────
    print_summary_table(all_results, args.output_dir)


def print_summary_table(results: list[dict], output_dir: str) -> None:
    from collections import defaultdict
    policy_stats = defaultdict(lambda: {
        "ratios": [], "nll": [], "saved_mb": [], "times": []
    })

    for r in results:
        if "error" in r:
            continue
        p = r["policy"]
        policy_stats[p]["ratios"].append(r["compression_ratio"])
        policy_stats[p]["nll"].append(r["continuation_nll"])
        policy_stats[p]["saved_mb"].append(r["memory_saved_mb"])
        policy_stats[p]["times"].append(r["total_seconds"])

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    header = (
        f"\n{'═'*85}\n"
        f"  {'POLICY':<16} {'AVG RATIO':>10} {'MEM SAVED':>10} "
        f"{'AVG NLL':>10} {'AVG PPL':>10} {'AVG TIME':>10}\n"
        f"{'─'*85}"
    )
    print(header)

    policy_order = ["recency","attention","hybrid","layerwise","sink","h2o","quant","scissor","learned"]
    rows = []
    for pol in policy_order:
        s = policy_stats.get(pol)
        if not s or not s["ratios"]:
            continue
        avg_ratio = avg(s["ratios"])
        avg_nll   = avg(s["nll"])
        avg_mb    = avg(s["saved_mb"])
        avg_time  = avg(s["times"])
        avg_ppl   = round(2**avg_nll, 2)
        star = " ⭐" if pol == "learned" else ""
        row = (
            f"  {pol.upper()+star:<18} {avg_ratio*100:>8.1f}%  {avg_mb:>8.2f} MB  "
            f"{avg_nll:>10.5f}  {avg_ppl:>10.2f}  {avg_time:>8.2f}s"
        )
        print(row)
        rows.append(row)

    footer = "═"*85
    print(footer)
    print("\n  Lower NLL/PPL = better quality preservation")
    print("  Higher ratio  = more compression (fewer tokens kept)")
    print("  ⭐ = Phase 5 Learned Memory\n")

    # Save table to file
    table_path = os.path.join(output_dir, "summary_table.txt")
    with open(table_path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
        f.write(footer + "\n\n")
        f.write("Lower NLL/PPL = better quality. Higher ratio = more compression.\n")
    print(f"✅ Summary table saved to: {table_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare all 9 KV-cache compression policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--n-prompts", type=int, default=3,
                   help="Number of prompts to use (1-10)")
    p.add_argument("--keep-last", type=int, default=256)
    p.add_argument("--keep-top", type=int, default=64)
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--n-memory", type=int, default=32)
    p.add_argument("--optim-steps", type=int, default=30)
    p.add_argument("--memory-lr", type=float, default=1e-2)
    p.add_argument("--keep-last-learned", type=int, default=64)
    p.add_argument("--dtype", default="float16",
                   choices=["float16","bfloat16","float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--output-dir", default="experiments/results/phase5")
    return p


if __name__ == "__main__":
    run_comparison(build_parser().parse_args())
