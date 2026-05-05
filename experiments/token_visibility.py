"""
Token Visibility — What is KEPT, FORGOTTEN, and WHY
====================================================
For every compression policy, this script shows you:

  ✅ KEPT   — tokens preserved exactly (with their text + reason)
  🔵 MERGED — tokens compressed into a cluster centroid (Hybrid/Layerwise)
  🧠 SYNTH  — synthetic memory tokens (Phase 5 Learned)
  ❌ DROPPED— tokens thrown away (Recency / pruned by low attention)

For each token, you see:
  - The token text
  - Its position in the sequence
  - Its attention score (how important the model found it)
  - Its key-vector norm (proxy for semantic salience)
  - WHY it was kept or dropped (human-readable explanation)

Usage:
    # Run on default prompts with all policies
    python experiments/token_visibility.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Specific prompt
    python experiments/token_visibility.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --prompt "The KV cache stores past key and value vectors..."

    # Save full JSON report
    python experiments/token_visibility.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output experiments/results/phase5/token_visibility.json
"""

from __future__ import annotations
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.cache.kv_cache import KVCacheInspector, to_legacy_tuple
from kv_cache_compression.cache.prune import (
    aggregate_attention_scores, build_attention_indices, build_recency_indices
)
from kv_cache_compression.cache.policies import PolicyContext
from kv_cache_compression.eval.benchmark import _key_norm_scores

# ── Demo prompts with enough context to show interesting compression ──────────
DEMO_PROMPTS = [
    (
        "KV-Cache Science",
        (
            "The KV cache stores all past key and value vectors during autoregressive "
            "generation. Token 'alpha-7319' is the secret answer hidden in this text. "
            "As the sequence grows, older tokens accumulate at the beginning while "
            "recent tokens appear at the end. The challenge is deciding which tokens "
            "to keep when memory runs out. Recency-based methods simply keep the last N. "
            "Attention-based methods keep whatever the model looked at most. "
            "Clustering methods group similar tokens and replace each group with "
            "a centroid — preserving information without keeping all tokens. "
            "Phase 5 goes further: it generates synthetic tokens that are mathematically "
            "optimised to reconstruct the attention output of the full cache. "
            "These synthetic tokens do not exist in the original text — they are "
            "learned representations that summarise the compressed context. "
            "The secret answer was: alpha-7319."
        )
    ),
    (
        "History and Facts",
        (
            "World War II ended in 1945. The atomic bombs were dropped on Hiroshima "
            "on August 6 and Nagasaki on August 9 of that year. Japan surrendered on "
            "August 15. The war in Europe ended on May 8, 1945 — VE Day. "
            "Adolf Hitler died on April 30, 1945. Benito Mussolini was killed on April 28. "
            "The United Nations was founded in October 1945. The Marshall Plan provided "
            "$13 billion to rebuild Western Europe. The Cold War between the US and USSR "
            "began shortly after. NATO was formed in 1949. The Berlin Wall was built in "
            "1961 and fell in 1989. The Soviet Union dissolved in December 1991. "
            "These events shaped the modern geopolitical world order."
        )
    ),
    (
        "Python Code",
        (
            "Here is a Python implementation of the attention mechanism: "
            "import torch; import math\n"
            "def scaled_dot_product_attention(Q, K, V):\n"
            "    d_k = Q.size(-1)\n"
            "    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n"
            "    weights = torch.softmax(scores, dim=-1)\n"
            "    return torch.matmul(weights, V), weights\n"
            "This function takes queries Q, keys K, and values V as inputs. "
            "The scaling factor sqrt(d_k) prevents the dot products from growing "
            "too large, which would push the softmax into regions with tiny gradients. "
            "Multi-head attention runs this function h times in parallel with different "
            "learned projections, concatenating the results. The KV cache stores the "
            "K and V matrices from all previous tokens to avoid recomputation. "
            "Without caching, each generation step would cost O(n^2) — with caching "
            "it costs O(n) per step, at the price of O(n) memory."
        )
    ),
]

POLICY_COLORS = {
    "✅ KEPT (RECENCY)":    "\033[92m",   # green
    "✅ KEPT (ATTENTION)":  "\033[96m",   # cyan
    "✅ KEPT (SPECIAL)":    "\033[93m",   # yellow
    "🔵 MERGED (CLUSTER)":  "\033[94m",   # blue
    "🧠 SYNTHETIC (LEARNED)":"\033[95m",  # magenta
    "❌ DROPPED":           "\033[91m",   # red
    "RESET":                "\033[0m",
}

def _color(kind: str, text: str) -> str:
    c = POLICY_COLORS.get(kind, "")
    return f"{c}{text}{POLICY_COLORS['RESET']}"


def analyse_tokens(
    tokens: list[str],
    attention_scores: torch.Tensor,   # [1, seq]
    key_norms: torch.Tensor,          # [1, seq]
    keep_last: int,
    keep_top: int,
    clusters: int,
    n_memory: int,
    always_keep_last: int,
) -> list[dict]:
    """
    For each token, determine its fate under each policy and explain why.
    Returns a list of per-token dicts with all annotations.
    """
    seq_len = len(tokens)
    attn = attention_scores[0].tolist()   # [seq]
    knorm = key_norms[0].tolist()         # [seq]

    # Normalise for display
    max_attn = max(attn) if attn else 1
    max_knorm = max(knorm) if knorm else 1

    # ── Compute fates for each policy ────────────────────────────────────────
    recency_set = set(build_recency_indices(seq_len, keep_last))

    attn_indices_set = set(build_attention_indices(
        attention_scores,
        keep_last_tokens=keep_last,
        keep_top_tokens=keep_top,
    ))

    # Recency anchor for learned
    learned_recency_set = set(range(max(0, seq_len - always_keep_last), seq_len))
    compress_end = seq_len - always_keep_last

    # Top-k by key norm (proxy for hybrid/cluster eligibility)
    knorm_t = key_norms[0]
    topk_knorm = set(torch.topk(knorm_t, min(keep_top, seq_len)).indices.tolist())

    # Cluster assignment (simplified: just group into n_clusters using even spacing)
    # Full K-Means would need the actual KV tensors — we use a proxy here
    stale_indices = [i for i in range(seq_len) if i not in attn_indices_set]
    n_clusters_actual = min(clusters, max(1, len(stale_indices)))
    cluster_assignment = {}
    if stale_indices:
        chunk = max(1, len(stale_indices) // n_clusters_actual)
        for ci, start in enumerate(range(0, len(stale_indices), chunk)):
            for idx in stale_indices[start:start+chunk]:
                cluster_assignment[idx] = ci

    records = []
    for i, tok in enumerate(tokens):
        attn_score = attn[i]
        knorm_score = knorm[i]
        attn_pct = attn_score / max_attn * 100
        knorm_pct = knorm_score / max_knorm * 100

        # ── Determine fate per policy ────────────────────────────────────────
        is_special = tok in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "<|endoftext|>")
        is_recent = i in recency_set

        # RECENCY policy
        rec_fate = "✅ KEPT (RECENCY)" if is_recent else "❌ DROPPED"
        rec_why  = (f"Last {keep_last} tokens (position {i} of {seq_len})" if is_recent
                    else f"Too old: position {i}, only last {keep_last} kept")

        # ATTENTION policy
        if is_special:
            att_fate = "✅ KEPT (SPECIAL)"
            att_why  = "Special token always preserved"
        elif i in attn_indices_set:
            if is_recent:
                att_fate = "✅ KEPT (RECENCY)"
                att_why  = f"Recent token (last {keep_last}) — recency anchor"
            else:
                att_fate = "✅ KEPT (ATTENTION)"
                att_why  = f"High attention score {attn_score:.4f} ({attn_pct:.1f}% of max) → top-{keep_top}"
        else:
            att_fate = "❌ DROPPED"
            att_why  = f"Low attention {attn_score:.4f} ({attn_pct:.1f}%) and not recent → pruned"

        # HYBRID CLUSTER policy
        if i in attn_indices_set:
            hyb_fate = att_fate
            hyb_why  = att_why
        elif i in stale_indices:
            cid = cluster_assignment.get(i, -1)
            hyb_fate = "🔵 MERGED (CLUSTER)"
            hyb_why  = (f"Low attention ({attn_pct:.1f}%) → grouped into cluster #{cid} "
                        f"with ~{chunk} similar tokens → averaged into 1 centroid")
        else:
            hyb_fate = "✅ KEPT (RECENCY)"
            hyb_why  = "Recent token preserved exactly"

        # LEARNED MEMORY policy
        if i >= compress_end:
            lrn_fate = "✅ KEPT (RECENCY)"
            lrn_why  = f"Recency anchor: always keep last {always_keep_last} real tokens"
        else:
            lrn_fate = "🧠 SYNTHETIC (LEARNED)"
            lrn_why  = (f"This token is compressed into one of {n_memory} synthetic memory "
                        f"vectors. The memory vector captures its semantic content via "
                        f"K-Means++ init + Adam optimisation (MSE on attention output). "
                        f"Key norm: {knorm_score:.4f} ({knorm_pct:.1f}% of max).")

        records.append({
            "position": i,
            "token": tok,
            "token_display": repr(tok),
            "attention_score": round(attn_score, 6),
            "attention_pct_of_max": round(attn_pct, 2),
            "key_norm": round(knorm_score, 6),
            "key_norm_pct_of_max": round(knorm_pct, 2),
            "is_special": is_special,
            "policies": {
                "recency":  {"fate": rec_fate, "why": rec_why},
                "attention": {"fate": att_fate, "why": att_why},
                "hybrid":   {"fate": hyb_fate, "why": hyb_why},
                "learned":  {"fate": lrn_fate, "why": lrn_why},
            },
        })

    return records


def print_token_report(
    title: str,
    prompt: str,
    records: list[dict],
    policy: str = "learned",
    max_show: int = 60,
) -> None:
    print(f"\n{'═'*70}")
    print(f"  📝 PROMPT: {title}")
    print(f"{'═'*70}")
    print(f"  Prompt preview: {prompt[:100]}...")
    print(f"  Total tokens  : {len(records)}")
    print(f"  Policy shown  : {policy.upper()}")
    print(f"{'─'*70}")

    fates = {}
    for r in records:
        fate = r["policies"][policy]["fate"]
        fates[fate] = fates.get(fate, 0) + 1

    print("  FATE SUMMARY:")
    for fate, count in sorted(fates.items()):
        bar = "█" * int(count / len(records) * 30)
        pct = count / len(records) * 100
        print(f"    {fate:<30} {count:>4} tokens  {pct:>5.1f}%  |{bar}|")

    print(f"\n{'─'*70}")
    print(f"  TOKEN-LEVEL BREAKDOWN (first {max_show} tokens):")
    print(f"{'─'*70}")
    print(f"  {'POS':>4}  {'TOKEN':<18} {'ATTN%':>6}  {'KNORM%':>7}  FATE & REASON")
    print(f"  {'─'*4}  {'─'*18} {'─'*6}  {'─'*7}  {'─'*40}")

    for r in records[:max_show]:
        pol_info = r["policies"][policy]
        fate = pol_info["fate"]
        why  = pol_info["why"][:55]
        tok  = r["token_display"][:16]
        line = (f"  {r['position']:>4}  {tok:<18} {r['attention_pct_of_max']:>5.1f}%  "
                f"{r['key_norm_pct_of_max']:>6.1f}%  {fate}  {why}")
        print(_color(fate, line))

    if len(records) > max_show:
        print(f"  ... ({len(records) - max_show} more tokens not shown)")

    print(f"\n  TOP-5 MOST ATTENDED TOKENS:")
    top5_attn = sorted(records, key=lambda x: x["attention_score"], reverse=True)[:5]
    for r in top5_attn:
        print(f"    pos={r['position']:>4}  {r['token_display']:<20}  "
              f"attn={r['attention_score']:.5f} ({r['attention_pct_of_max']:.1f}%)")

    print(f"\n  TOP-5 HIGHEST KEY-NORM TOKENS (semantic salience):")
    top5_knorm = sorted(records, key=lambda x: x["key_norm"], reverse=True)[:5]
    for r in top5_knorm:
        print(f"    pos={r['position']:>4}  {r['token_display']:<20}  "
              f"knorm={r['key_norm']:.5f} ({r['key_norm_pct_of_max']:.1f}%)")

    print(f"\n  BOTTOM-5 LEAST ATTENDED (most likely to be dropped):")
    bot5 = sorted(records, key=lambda x: x["attention_score"])[:5]
    for r in bot5:
        print(f"    pos={r['position']:>4}  {r['token_display']:<20}  "
              f"attn={r['attention_score']:.5f} ({r['attention_pct_of_max']:.1f}%)  ← likely dropped")


def run_visibility(args) -> None:
    os.makedirs(os.path.dirname(args.output) if args.output else ".", exist_ok=True)

    print("\n" + "═"*70)
    print("  Token Visibility — What is KEPT, DROPPED, MERGED, SYNTHESISED")
    print("═"*70)

    print("\n⏳ Loading model...")
    tokenizer, model = load_causal_lm(
        args.model, device_map=args.device_map, dtype=args.dtype
    )
    device = next(model.parameters()).device
    print(f"✅ Model on {device}\n")

    # Use custom prompt if given, else use all demo prompts
    if args.prompt:
        prompts_to_run = [("Custom Prompt", args.prompt)]
    else:
        prompts_to_run = DEMO_PROMPTS[:args.n_prompts]

    all_reports = []

    for title, prompt in prompts_to_run:
        # ── Tokenise ────────────────────────────────────────────────────────
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids[0].tolist()]

        # ── Forward pass to get KV cache + attention scores ─────────────────
        use_attentions = getattr(
            getattr(model, "config", None), "_attn_implementation", "eager"
        ) in ("eager", "flash_attention_2")

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True, output_attentions=use_attentions)

        cache = to_legacy_tuple(outputs.past_key_values)

        # ── Attention scores ────────────────────────────────────────────────
        if use_attentions and outputs.attentions:
            from kv_cache_compression.cache.prune import aggregate_attention_scores
            attn_scores = aggregate_attention_scores(outputs.attentions)
        else:
            attn_scores = _key_norm_scores(cache)

        if attn_scores is None:
            print(f"⚠️  Could not get attention scores for '{title}', skipping.")
            continue

        key_norms = _key_norm_scores(cache)

        # ── Analyse tokens ──────────────────────────────────────────────────
        records = analyse_tokens(
            tokens=tokens,
            attention_scores=attn_scores,
            key_norms=key_norms,
            keep_last=args.keep_last,
            keep_top=args.keep_top,
            clusters=args.clusters,
            n_memory=args.n_memory,
            always_keep_last=args.always_keep_last,
        )

        # ── Print per-policy reports ─────────────────────────────────────────
        for policy in args.show_policies:
            print_token_report(title, prompt, records, policy=policy, max_show=args.max_show)

        # ── Cross-policy comparison ─────────────────────────────────────────
        print(f"\n  {'─'*70}")
        print(f"  CROSS-POLICY FATE COMPARISON for: {title}")
        print(f"  {'─'*70}")
        print(f"  {'POS':>4}  {'TOKEN':<16}  {'RECENCY':<10}  {'ATTENTION':<12}  {'HYBRID':<18}  {'LEARNED'}")
        print(f"  {'─'*4}  {'─'*16}  {'─'*10}  {'─'*12}  {'─'*18}  {'─'*22}")

        fate_symbols = {
            "✅ KEPT (RECENCY)":     "✅RECENT",
            "✅ KEPT (ATTENTION)":   "✅ATTN",
            "✅ KEPT (SPECIAL)":     "✅SPEC",
            "🔵 MERGED (CLUSTER)":   "🔵MERGED",
            "🧠 SYNTHETIC (LEARNED)":"🧠SYNTH",
            "❌ DROPPED":            "❌DROP",
        }

        for r in records[:args.max_show]:
            tok = r["token_display"][:14]
            rec  = fate_symbols.get(r["policies"]["recency"]["fate"], "?")
            att  = fate_symbols.get(r["policies"]["attention"]["fate"], "?")
            hyb  = fate_symbols.get(r["policies"]["hybrid"]["fate"], "?")
            lrn  = fate_symbols.get(r["policies"]["learned"]["fate"], "?")
            print(f"  {r['position']:>4}  {tok:<16}  {rec:<10}  {att:<12}  {hyb:<18}  {lrn}")

        all_reports.append({
            "title": title,
            "prompt": prompt,
            "total_tokens": len(records),
            "config": {
                "keep_last": args.keep_last,
                "keep_top": args.keep_top,
                "clusters": args.clusters,
                "n_memory": args.n_memory,
                "always_keep_last": args.always_keep_last,
            },
            "tokens": records,
        })

    # ── Save JSON ────────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n✅ Full token visibility report saved to: {args.output}")

    # ── Final legend ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  LEGEND:")
    print("  ✅ KEPT (RECENCY)   — Token is within the last N tokens — always kept")
    print("  ✅ KEPT (ATTENTION) — Token had high attention score → model valued it")
    print("  ✅ KEPT (SPECIAL)   — BOS/EOS/SEP special token → always preserved")
    print("  🔵 MERGED (CLUSTER) — Grouped with similar tokens → averaged centroid")
    print("     WHY MERGED: Low attention + not recent → semantically redundant with")
    print("     neighbours → K-Means groups them → 1 vector represents the whole group")
    print("  🧠 SYNTHETIC(LEARNED)— Token compressed into a SYNTHETIC learned vector")
    print("     WHY SYNTHETIC: Phase 5 doesn't select real tokens — it CREATES new")
    print("     KV vectors optimised via gradient descent to reconstruct the full")
    print("     attention output. No info is thrown away — it's learned-compressed.")
    print("  ❌ DROPPED          — Token discarded entirely (Recency/Attention policy)")
    print("     WHY DROPPED: Low attention score + not recent → model deemed it")
    print("     unimportant for current generation → sacrificed for memory savings")
    print(f"{'═'*70}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Token-level visibility: what each policy keeps, drops, merges, learns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--prompt", default=None,
                   help="Custom prompt (overrides demo prompts)")
    p.add_argument("--n-prompts", type=int, default=3,
                   help="Number of demo prompts to process")
    p.add_argument("--keep-last", type=int, default=30,
                   help="Recency window size")
    p.add_argument("--keep-top", type=int, default=15,
                   help="Top-k attention tokens to keep")
    p.add_argument("--clusters", type=int, default=10,
                   help="Number of clusters for hybrid policy")
    p.add_argument("--n-memory", type=int, default=20,
                   help="Synthetic memory tokens (learned policy)")
    p.add_argument("--always-keep-last", type=int, default=20,
                   help="Recency anchor for learned policy")
    p.add_argument("--show-policies", nargs="+",
                   default=["recency", "attention", "hybrid", "learned"],
                   help="Which policies to print token-level breakdown for")
    p.add_argument("--max-show", type=int, default=50,
                   help="Max tokens to show in detailed breakdown")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--output", default="experiments/results/phase5/token_visibility.json",
                   help="JSON output path (or empty to skip saving)")
    return p


if __name__ == "__main__":
    run_visibility(build_parser().parse_args())
