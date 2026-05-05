# KV-Cache Compression for Long-Context LLM Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.40+-yellow.svg)](https://huggingface.co/docs/transformers)

Research-grade toolkit for compressing the KV cache during LLM inference — reducing GPU memory usage by 50-90% while preserving generation quality. Works with any HuggingFace causal language model, no retraining required.

## The Problem

When an LLM processes a long prompt, it stores all past keys and values from every attention layer in the **KV cache**. This cache grows linearly with context length and becomes the dominant GPU memory bottleneck. For a 32-layer model at 8K tokens, the KV cache alone requires ~4 GB.

## What This Does

Implements and benchmarks **9 compression methods** that selectively remove, merge, or replace less important tokens in the KV cache during inference. Includes a full evaluation suite with built-in datasets for QA, summarization, and code completion tasks.

<pre class="no-copy">
┌─────────────────────────────────────────────────────────┐
│                   Full KV Cache                         │
│  [token_0] [token_1] [token_2] ... [token_N-1]         │
└─────────────────────────────────────────────────────────┘
                          │
                    Compression
                      Policy
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Keep Recent  Keep Important  Cluster/Merge
         Tokens       Tokens (attn)   Stale Tokens
              │           │           │
              └───────────┼───────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│               Compressed KV Cache                       │
│  [token_0] [token_N-64..N-1] [centroid_1] [centroid_2]  │
│  (sinks)   (recency)         (clustered summaries)      │
└─────────────────────────────────────────────────────────┘
</pre>

## Compression Policies

| Policy | Flag | Description | Best For |
|--------|------|-------------|----------|
| **Recency Window** | `--policy recency` | Keep only the last N tokens | Baseline / speed |
| **Attention Retention** | `--policy attention` | Recent + top-K highest-attention tokens | Quality-focused |
| **Hybrid Cluster** | `--policy hybrid` | Recent + attention + k-means clustering of stale tokens | Best tradeoff |
| **Layerwise Hybrid** | `--policy layerwise` | Per-layer budgets (heavier compression at bottom layers) | Deep models |
| **Sink Token** | `--policy sink` | Always keep first N "sink" tokens + sliding window | Streaming |
| **Heavy Hitter (H2O)** | `--policy h2o` | Track cumulative per-token attention, evict lowest | Attention-heavy |
| **Quantized Cache** | `--policy quantized` | INT8 quantization of KV tensors (~50% memory reduction) | No eviction |
| **ScissorHands** | `--policy scissorhands` | Unified: sinks + heavy hitters + recency window | Balanced |
| **Learned Memory** | `--policy learned` | Replace stale tokens with optimized synthetic memory vectors | Research |

### Budget Schedulers

Control how compression aggressiveness changes over the sequence:

- `--scheduler linear` — Linear interpolation from mild to aggressive
- `--scheduler exponential` — Exponential decay of budget ratio
- `--scheduler step` — Discrete step function with thresholds
- `--scheduler memory-aware` — Adjusts based on available GPU memory

## Quick Start

### Install

```bash
git clone https://github.com/EurydiceReverie/Adaptive-KV-Cache-management.git
cd Adaptive-KV-Cache-management
pip install -r requirements.txt
```

### Run a Single Policy

```bash
# Recency baseline (keep last 512 tokens)
python main.py --policy recency --keep-last 512

# Attention-aware pruning
python main.py --policy attention --keep-last 512 --keep-top 128

# Hybrid clustering (best quality/compression tradeoff)
python main.py --policy hybrid --keep-last 512 --keep-top 128 --clusters 64

# Learned memory tokens (32 synthetic + 64 recency anchors)
python main.py --policy learned --n-memory 32 --memory-optim-steps 50 --memory-keep-last 64
```

### Run Full Comparison

```bash
# Compare all policies on needle-in-haystack + QA tasks
python experiments/compare_policies.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --keep-last 512 --keep-top 128 --clusters 64

# Evaluate all 8 policies across QA, summarization, and code completion
python experiments/advanced_eval.py

# Phase 5: learned memory vs all other policies
python experiments/phase5_compare_all.py
```

### Generate Plots

```python
from kv_cache_compression.utils.plotting import generate_all_plots
generate_all_plots("experiments/results/results.json", output_dir="experiments/results/plots")
```

Results are generated locally and saved to `experiments/results/` (gitignored).

## Evaluation Suite

Built-in datasets — no internet required after model download:

| Task | Evaluator | Samples | Metrics |
|------|-----------|---------|---------|
| **Long-context QA** | `long_qa_eval.py` | 15 multi-document samples | Token-F1, Exact Match |
| **Summarization** | `summarization_eval.py` | 10 articles | ROUGE-1/2/L, Faithfulness |
| **Code Completion** | `code_completion_eval.py` | 12 Python + 9 JS/SQL/Bash | Syntax validity, Edit distance |
| **Needle-in-Haystack** | `needle_eval.py` | Configurable | Needle recall |
| **Perplexity** | `perplexity_eval.py` | Any text | NLL, Perplexity |

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| Compression ratio | compressed / original tokens |
| Tokens saved % | Percentage of cache eliminated |
| Memory bytes | GPU memory before/after |
| Continuation NLL | Negative log-likelihood (lower = better) |
| Perplexity | exp(NLL) — generation quality |
| Prompt latency | Forward pass time |
| Compression latency | Time spent compressing |

## Models

Works with any HuggingFace causal LM. Default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

**Models are not included in this repo.** Download before running:

```bash
# Download default model (~2 GB)
python download_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Or download any HuggingFace causal LM
python download_model.py --model Qwen/Qwen2-1.5B-Instruct
python download_model.py --model mistralai/Mistral-7B-v0.1
```

Models are saved to `models/` (gitignored).

| Model | VRAM | Notes |
|-------|------|-------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2 GB | Fast iteration |
| `Qwen/Qwen2-1.5B-Instruct` | ~3 GB | Strong small model |
| `mistralai/Mistral-7B-v0.1` | ~14 GB | Research scale |
| `meta-llama/Llama-2-7b-hf` | ~14 GB | Industry standard |

## Example Output

<pre class="no-copy">
━━━ Results: Needle-in-Haystack ━━━
Policy                  Orig Tok   Comp Tok    Saved%    Orig Mem    Comp Mem       NLL       PPL
no_compression              4096       4096      0.0%    128.0 MB    128.0 MB    2.1234    8.36
recency_window              4096        512     87.5%    128.0 MB     16.0 MB    3.4521   31.62
attention_retention         4096        640     84.4%    128.0 MB     20.0 MB    2.5100   12.31
hybrid_cluster              4096        704     82.8%    128.0 MB     22.0 MB    2.2340    9.33
</pre>

## Key Design Decisions

- **No retraining** — all methods work at inference time with any frozen HuggingFace model
- **Modular** — add new policies by subclassing `CompressionPolicy`
- **Batch size 1** — targets single-sequence long-context inference
- **Float16 default** — works on consumer GPUs with 8GB+ VRAM

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Transformers 4.40+
- 8GB+ VRAM (with TinyLlama)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.