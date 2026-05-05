"""
Inspect exactly what happens during KV-cache compression.

Shows:
  - The full prompt with each token labeled: [KEPT] [DROPPED] [CLUSTER-N]
  - A summary of recency vs attention-kept vs clustered tokens
  - Which tokens were most attended to
  - Cluster membership details

Usage:
    python experiments/inspect_compression.py --policy hybrid
    python experiments/inspect_compression.py --policy attention
    python experiments/inspect_compression.py --policy recency
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kv_cache_compression.cache.cluster import HybridClusterPolicy, _simple_kmeans, _token_features_from_cache
from kv_cache_compression.cache.prune import (
    AttentionRetentionPolicy, RecencyWindowPolicy,
    build_attention_indices, build_recency_indices,
    aggregate_attention_scores,
)
from kv_cache_compression.cache.kv_cache import to_legacy_tuple, KVCacheInspector
from kv_cache_compression.eval.benchmark import _key_norm_scores
from kv_cache_compression.models.model_loader import load_causal_lm
from kv_cache_compression.utils.profiling import format_bytes


COLORS = {
    "KEPT_RECENT":    "\033[92m",   # green
    "KEPT_ATTENTION": "\033[94m",   # blue
    "CLUSTERED":      "\033[93m",   # yellow
    "DROPPED":        "\033[91m",   # red
    "RESET":          "\033[0m",
    "BOLD":           "\033[1m",
    "CYAN":           "\033[96m",
    "MAGENTA":        "\033[95m",
}

def colored(text, color_key):
    return f"{COLORS.get(color_key, '')}{text}{COLORS['RESET']}"


def build_parser():
    p = argparse.ArgumentParser(description="Inspect KV-cache compression token decisions")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--policy", choices=["recency", "attention", "hybrid"], default="hybrid")
    p.add_argument("--keep-last", type=int, default=512)
    p.add_argument("--keep-top", type=int, default=128)
    p.add_argument("--clusters", type=int, default=64)
    p.add_argument("--prompt", type=str, default=None, help="Custom prompt (default: needle haystack)")
    p.add_argument("--needle", type=str, default="alpha-7319")
    p.add_argument("--haystack-repeats", type=int, default=80)
    p.add_argument("--no-color", action="store_true", help="Disable terminal colors")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--show-top-attended", type=int, default=20, help="Show N most attended tokens")
    return p


def make_prompt(needle: str, repeats: int) -> str:
    filler = (
        f"The quick brown fox jumps over the lazy dog. "
        f"Some information may be useful and some may not. "
        f"Keep reading to find the important data. "
    )
    haystack = filler * repeats
    return (
        f"Please read the following document carefully.\n\n"
        f"{haystack}\n\n"
        f"IMPORTANT: The secret code is {needle}.\n\n"
        f"{filler * (repeats // 4)}\n\n"
        f"What is the secret code?"
    )


def get_max_length(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return 2048


def run_inspection(args):
    no_color = args.no_color

    def c(text, key):
        return text if no_color else colored(text, key)

    print(c("\n══════════════════════════════════════════════════════", "BOLD"))
    print(c("   KV-Cache Compression Inspector", "BOLD"))
    print(c("══════════════════════════════════════════════════════\n", "BOLD"))

    # ── Load model ──
    print(f"Loading model: {args.model}")
    tokenizer, model = load_causal_lm(args.model, device_map="auto", dtype="float16")
    device = next(model.parameters()).device
    max_len = get_max_length(model)

    # ── Build prompt ──
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = make_prompt(args.needle, args.haystack_repeats)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > max_len:
        print(f"[inspector] Truncating prompt: {input_ids.shape[1]} → {max_len} tokens")
        input_ids = input_ids[:, -max_len:]
    input_ids = input_ids.to(device)

    seq_len = input_ids.shape[1]
    tokens_text = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    print(f"\n{c('Prompt stats:', 'BOLD')}")
    print(f"  Total tokens : {seq_len}")
    print(f"  Model max    : {max_len}")
    print(f"  Policy       : {args.policy}")
    print(f"  Keep last    : {args.keep_last}")
    if args.policy in ("attention", "hybrid"):
        print(f"  Keep top-attn: {args.keep_top}")
    if args.policy == "hybrid":
        print(f"  Clusters     : {args.clusters}")

    # ── Forward pass ──
    print(f"\n{c('Running forward pass...', 'CYAN')}")
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", "eager")
    use_attentions = attn_impl in ("eager", "flash_attention_2")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True, output_attentions=use_attentions)

    past_kv = to_legacy_tuple(outputs.past_key_values)

    # ── Attention scores ──
    if use_attentions and outputs.attentions:
        attn_scores = aggregate_attention_scores(outputs.attentions)  # [1, seq_len]
    else:
        print(f"  [{c('note', 'CYAN')}] output_attentions not supported ({attn_impl}), using key-norm scores")
        attn_scores = _key_norm_scores(past_kv)  # [1, seq_len]

    attn_scores_flat = attn_scores[0].cpu()  # [seq_len]

    # ── Compute kept/dropped/clustered indices per policy ──
    recency_set = set(build_recency_indices(seq_len, args.keep_last))

    if args.policy == "recency":
        kept_set = recency_set
        attention_kept_set = set()
        cluster_assignments = {}   # token_idx → cluster_id
        stale_set = set(range(seq_len)) - kept_set

    elif args.policy == "attention":
        kept_indices = build_attention_indices(
            attn_scores, keep_last_tokens=args.keep_last, keep_top_tokens=args.keep_top
        )
        kept_set = set(kept_indices)
        attention_kept_set = kept_set - recency_set
        cluster_assignments = {}
        stale_set = set(range(seq_len)) - kept_set

    else:  # hybrid
        kept_indices = build_attention_indices(
            attn_scores, keep_last_tokens=args.keep_last, keep_top_tokens=args.keep_top
        )
        exact_keep = set(kept_indices)
        attention_kept_set = exact_keep - recency_set
        stale_indices = [i for i in range(seq_len) if i not in exact_keep]

        # Run clustering on stale tokens
        num_clusters = min(args.clusters, len(stale_indices))
        features = _token_features_from_cache(past_kv)
        if len(stale_indices) > 0 and num_clusters > 0:
            stale_features = features[stale_indices]
            assignments_tensor = _simple_kmeans(stale_features.unsqueeze(0).expand(len(stale_indices), -1)
                                                if stale_features.dim() == 1 else stale_features,
                                                num_clusters=num_clusters)
            cluster_assignments = {stale_indices[i]: int(assignments_tensor[i].item())
                                   for i in range(len(stale_indices))}
        else:
            cluster_assignments = {}

        kept_set = exact_keep
        stale_set = set(range(seq_len)) - exact_keep - set(cluster_assignments.keys())

    # ── Token-level label assignment ──
    labels = []
    for i in range(seq_len):
        if i in recency_set and args.policy != "recency":
            labels.append(("KEPT_RECENT", f"RECENT"))
        elif i in kept_set and i not in recency_set:
            labels.append(("KEPT_ATTENTION", f"ATTN"))
        elif i in kept_set:
            labels.append(("KEPT_RECENT", f"RECENT"))
        elif i in cluster_assignments:
            labels.append(("CLUSTERED", f"C{cluster_assignments[i]}"))
        else:
            labels.append(("DROPPED", "DROP"))

    # ── Summary stats ──
    n_recent  = sum(1 for l, _ in labels if l == "KEPT_RECENT")
    n_attn    = sum(1 for l, _ in labels if l == "KEPT_ATTENTION")
    n_cluster = sum(1 for l, _ in labels if l == "CLUSTERED")
    n_dropped = sum(1 for l, _ in labels if l == "DROPPED")
    n_kept    = n_recent + n_attn + (n_cluster if args.policy == "hybrid" else 0)

    print(f"\n{c('═══ Compression Summary ═══', 'BOLD')}")
    print(f"  Original tokens  : {c(str(seq_len), 'BOLD')}")
    print(f"  {c('KEPT (recency)', 'KEPT_RECENT')}   : {n_recent:>5}  ({100*n_recent/seq_len:.1f}%)")
    if args.policy in ("attention", "hybrid"):
        print(f"  {c('KEPT (attention)', 'KEPT_ATTENTION')} : {n_attn:>5}  ({100*n_attn/seq_len:.1f}%)")
    if args.policy == "hybrid":
        print(f"  {c('CLUSTERED', 'CLUSTERED')}        : {n_cluster:>5}  ({100*n_cluster/seq_len:.1f}%)  → compressed to {args.clusters} representatives")
    print(f"  {c('DROPPED', 'DROPPED')}          : {n_dropped:>5}  ({100*n_dropped/seq_len:.1f}%)")
    print(f"  Effective cache  : {c(str(seq_len - n_dropped - n_cluster + (args.clusters if args.policy=='hybrid' else 0)), 'BOLD')} tokens")

    mem_original = KVCacheInspector.total_bytes(past_kv)
    ratio = (n_kept + (args.clusters if args.policy == "hybrid" else 0)) / seq_len
    mem_compressed = int(mem_original * ratio)
    print(f"  Memory: {format_bytes(mem_original)} → ~{format_bytes(mem_compressed)} "
          f"({100*(1-ratio):.1f}% saved)")

    # ── Top attended tokens ──
    top_k = min(args.show_top_attended, seq_len)
    top_indices = torch.topk(attn_scores_flat, k=top_k).indices.tolist()
    print(f"\n{c(f'═══ Top {top_k} Most Attended Tokens ═══', 'BOLD')}")
    print(f"  {'Pos':>5}  {'Score':>8}  {'Label':<10}  Token")
    print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*30}")
    for rank, idx in enumerate(top_indices):
        score = float(attn_scores_flat[idx])
        label_key, label_name = labels[idx]
        tok = tokens_text[idx] if idx < len(tokens_text) else "?"
        tok_display = tok.replace("▁", "_").replace("\n", "\\n")[:30]
        print(f"  {idx:>5}  {score:>8.5f}  {c(f'{label_name:<10}', label_key)}  {tok_display}")

    # ── Token-by-token display (chunked) ──
    print(f"\n{c('═══ Full Token Map ═══', 'BOLD')}")
    legend = (
        f"  Legend: "
        f"{c('RECENT', 'KEPT_RECENT')} = kept (recency)  "
        f"{c('ATTN', 'KEPT_ATTENTION')} = kept (attention)  "
        f"{c('C#', 'CLUSTERED')} = clustered  "
        f"{c('DROP', 'DROPPED')} = discarded"
    )
    print(legend)
    print()

    chunk_size = 10
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        line_tokens = []
        line_labels = []
        for i in range(start, end):
            tok = tokens_text[i] if i < len(tokens_text) else "?"
            tok = tok.replace("▁", "_").replace("\n", "\\n")[:12]
            label_key, label_name = labels[i]
            line_tokens.append(f"{tok:<12}")
            line_labels.append(c(f"{label_name:<12}", label_key))

        pos_line   = f"  [{start:>4}]  " + " ".join(f"{i:<12}" for i in range(start, end))
        token_line = f"  {'token':>6}   " + " ".join(line_tokens)
        label_line = f"  {'label':>6}   " + " ".join(line_labels)
        print(pos_line)
        print(token_line)
        print(label_line)
        print()

    # ── Cluster details (hybrid only) ──
    if args.policy == "hybrid" and cluster_assignments:
        print(f"\n{c('═══ Cluster Details ═══', 'BOLD')}")
        from collections import defaultdict
        clusters: dict[int, list[int]] = defaultdict(list)
        for tok_idx, cluster_id in cluster_assignments.items():
            clusters[cluster_id].append(tok_idx)

        for cid in sorted(clusters.keys())[:20]:  # show first 20 clusters
            members = sorted(clusters[cid])
            member_tokens = [
                tokens_text[i].replace("▁", "_").replace("\n", "\\n")[:10]
                if i < len(tokens_text) else "?"
                for i in members[:8]
            ]
            print(f"  {c(f'Cluster {cid:>3}', 'CLUSTERED')} ({len(members):>4} tokens)  "
                  f"pos: {members[0]}–{members[-1]}  "
                  f"tokens: [{', '.join(member_tokens)}{'...' if len(members)>8 else ''}]")

        if len(clusters) > 20:
            print(f"  ... and {len(clusters)-20} more clusters")

    # ── Save JSON ──
    if args.output_json:
        result = {
            "policy": args.policy,
            "seq_len": seq_len,
            "n_recent": n_recent,
            "n_attention": n_attn,
            "n_clustered": n_cluster,
            "n_dropped": n_dropped,
            "top_attended": [
                {
                    "position": idx,
                    "score": float(attn_scores_flat[idx]),
                    "token": tokens_text[idx] if idx < len(tokens_text) else "?",
                    "label": labels[idx][1],
                }
                for idx in top_indices
            ],
            "token_labels": [
                {
                    "position": i,
                    "token": tokens_text[i] if i < len(tokens_text) else "?",
                    "label": labels[i][1],
                    "attention_score": float(attn_scores_flat[i]),
                }
                for i in range(seq_len)
            ],
        }
        if args.policy == "hybrid":
            from collections import defaultdict
            clusters_dict = defaultdict(list)
            for tok_idx, cid in cluster_assignments.items():
                clusters_dict[cid].append(tok_idx)
            result["clusters"] = {
                str(cid): {
                    "member_positions": members,
                    "member_tokens": [
                        tokens_text[i].replace("▁", "_") if i < len(tokens_text) else "?"
                        for i in members
                    ],
                }
                for cid, members in clusters_dict.items()
            }

        os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n{c('Results saved to: ' + args.output_json, 'CYAN')}")

    print(f"\n{c('══════════════════════════════════════════════════════', 'BOLD')}\n")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_inspection(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
