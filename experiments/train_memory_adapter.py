"""
Phase 5 — Train the MemoryTokenAdapter (offline, one-time).

This script trains a lightweight MLP that learns to predict good initial
KV-vector memory tokens from compact cache statistics.  Once trained, the
adapter is saved to disk and used by `LearnedMemoryFineTunePolicy` as a
warm-start — reducing the number of inference-time optimisation steps from
~50 (no adapter) down to ~5-10.

Training overview
-----------------
1. Load a causal LM and run forward passes on random / dataset prompts.
2. For each layer of each forward pass, collect (key, value) tensors.
3. Train the adapter to minimise the MSE between:
       attention_output(predicted_memory_tokens)
   vs  attention_output(real_tokens)
4. Save the adapter weights to `--output`.

Usage
-----
# Quick smoke-test (CPU, TinyLlama, 20 steps):
python experiments/train_memory_adapter.py \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --n-memory 32 \\
    --steps 20 \\
    --output experiments/results/memory_adapter.pt

# Full training run (GPU recommended):
python experiments/train_memory_adapter.py \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --n-memory 32 \\
    --steps 500 \\
    --lr 3e-4 \\
    --output experiments/results/memory_adapter.pt

After training, use the adapter:
    from kv_cache_compression.cache.learned_memory import (
        MemoryTokenAdapter, LearnedMemoryFineTunePolicy
    )
    adapter = MemoryTokenAdapter.load("experiments/results/memory_adapter.pt",
                                      head_dim=64, n_memory=32)
    policy = LearnedMemoryFineTunePolicy(adapter=adapter, n_memory=32, optim_steps=10)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from kv_cache_compression.cache.learned_memory import MemoryTokenAdapter
from kv_cache_compression.cache.kv_cache import to_legacy_tuple
from kv_cache_compression.models.model_loader import load_causal_lm


# ─── Synthetic training prompts (no external dataset needed) ─────────────────

_TRAIN_TEMPLATES = [
    "The history of {topic} spans many centuries. Scholars have debated its origins since {year}.",
    "In the field of {topic}, recent advances have led to significant breakthroughs in {year}.",
    "A detailed analysis of {topic} reveals that the core principles date back to {year}.",
    "Researchers studying {topic} have consistently found patterns that emerge after {year}.",
    "The evolution of {topic} from {year} to the present day illustrates key trends.",
    "Understanding {topic} requires examining the foundational work established in {year}.",
    "The relationship between {topic} and related fields has grown substantially since {year}.",
    "Many practitioners in {topic} cite {year} as a turning point in the discipline.",
    "A comprehensive review of {topic} literature from {year} onwards shows steady growth.",
    "The principles governing {topic} were first formalised around {year} by leading experts.",
]

_TOPICS = [
    "machine learning", "natural language processing", "computer vision",
    "distributed systems", "quantum computing", "bioinformatics",
    "reinforcement learning", "data compression", "cryptography",
    "graph theory", "signal processing", "robotics", "neuroscience",
    "climate modelling", "game theory", "financial mathematics",
]

_YEARS = [str(y) for y in range(1980, 2025)]


def _make_synthetic_prompt(rng: random.Random, min_words: int = 80) -> str:
    """Generate a random multi-sentence synthetic prompt."""
    parts = []
    while sum(len(p.split()) for p in parts) < min_words:
        tmpl = rng.choice(_TRAIN_TEMPLATES)
        topic = rng.choice(_TOPICS)
        year = rng.choice(_YEARS)
        parts.append(tmpl.format(topic=topic, year=year))
    return " ".join(parts)


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print(f"\n{'='*60}")
    print(f"  Phase 5 — MemoryTokenAdapter Training")
    print(f"{'='*60}")
    print(f"  Model     : {args.model}")
    print(f"  n_memory  : {args.n_memory}")
    print(f"  Steps     : {args.steps}")
    print(f"  LR        : {args.lr}")
    print(f"  Output    : {args.output}")
    print(f"{'='*60}\n")

    # ── Load model ──
    print("Loading model...")
    tokenizer, model = load_causal_lm(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}\n")

    # ── Discover head_dim from one forward pass ──
    rng = random.Random(42)
    probe_text = _make_synthetic_prompt(rng, min_words=30)
    probe_inputs = tokenizer(probe_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        probe_out = model(**probe_inputs, use_cache=True)
    probe_cache = to_legacy_tuple(probe_out.past_key_values)
    if not probe_cache:
        print("[ERROR] Model returned empty KV cache. Aborting.")
        sys.exit(1)

    _, num_heads, _, head_dim = probe_cache[0][0].shape
    num_layers = len(probe_cache)
    print(f"Model architecture: {num_layers} layers, {num_heads} heads, head_dim={head_dim}\n")

    # ── Build adapter (one per layer — shared weights, conditioned on stats) ──
    # We train a single shared adapter across all layers for simplicity.
    # For best quality, train per-layer; for speed, shared is sufficient.
    adapter = MemoryTokenAdapter(
        head_dim=head_dim,
        n_memory=args.n_memory,
        hidden_dim=args.hidden_dim,
        num_layers=args.adapter_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    loss_history = []
    best_loss = float("inf")
    best_state = None

    print(f"Starting training for {args.steps} steps...\n")

    for step in range(1, args.steps + 1):
        # ── Generate a fresh synthetic prompt ──
        prompt = _make_synthetic_prompt(rng, min_words=args.min_prompt_words)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            out = model(**inputs, use_cache=True)
        cache = to_legacy_tuple(out.past_key_values)
        if not cache:
            continue

        # ── Compute loss across a random sample of layers ──
        layer_indices = random.sample(range(num_layers), min(args.layers_per_step, num_layers))
        total_loss = torch.tensor(0.0, device=device)

        adapter.train()
        optimizer.zero_grad()

        for li in layer_indices:
            key, value = cache[li]
            key = key.to(device)
            value = value.to(device)
            loss = adapter.compute_loss(key, value)
            total_loss = total_loss + loss

        avg_loss = total_loss / len(layer_indices)
        avg_loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler_lr.step()

        loss_val = float(avg_loss.item())
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        if step % max(1, args.steps // 10) == 0 or step == 1:
            avg_recent = sum(loss_history[-10:]) / len(loss_history[-10:])
            print(f"  Step {step:>5}/{args.steps}  loss={loss_val:.6f}  "
                  f"avg(last10)={avg_recent:.6f}  best={best_loss:.6f}  "
                  f"lr={scheduler_lr.get_last_lr()[0]:.2e}")

    # ── Save best adapter ──
    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.save(args.output)

    # Save training metadata alongside weights
    meta_path = args.output.replace(".pt", "_meta.json")
    meta = {
        "model": args.model,
        "n_memory": args.n_memory,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_dim": args.hidden_dim,
        "adapter_layers": args.adapter_layers,
        "steps": args.steps,
        "best_loss": best_loss,
        "final_loss": loss_history[-1] if loss_history else None,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best loss     : {best_loss:.6f}")
    print(f"   Adapter saved : {args.output}")
    print(f"   Metadata      : {meta_path}")
    print(f"\nTo use the adapter:")
    print(f"    from kv_cache_compression.cache.learned_memory import (")
    print(f"        MemoryTokenAdapter, LearnedMemoryFineTunePolicy")
    print(f"    )")
    print(f"    adapter = MemoryTokenAdapter.load('{args.output}',")
    print(f"                                      head_dim={head_dim}, n_memory={args.n_memory})")
    print(f"    policy  = LearnedMemoryFineTunePolicy(adapter=adapter, n_memory={args.n_memory})")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MemoryTokenAdapter for Phase 5 KV-cache compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HuggingFace model name or local path")
    p.add_argument("--n-memory", type=int, default=32,
                   help="Number of memory tokens the adapter predicts")
    p.add_argument("--steps", type=int, default=200,
                   help="Total training steps")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="AdamW learning rate")
    p.add_argument("--hidden-dim", type=int, default=256,
                   help="Hidden dimension of adapter MLP")
    p.add_argument("--adapter-layers", type=int, default=2,
                   help="Number of hidden layers in adapter MLP")
    p.add_argument("--layers-per-step", type=int, default=4,
                   help="Number of transformer layers to sample per training step")
    p.add_argument("--min-prompt-words", type=int, default=80,
                   help="Minimum words in synthetic training prompts")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output", default="models/memory_adapter.pt",
                   help="Path to save the trained adapter weights (default: models/)")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
