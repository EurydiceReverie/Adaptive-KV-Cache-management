"""
download_model.py — Download and cache a HuggingFace model locally into models/

Usage:
    python download_model.py
    python download_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python download_model.py --model meta-llama/Llama-2-7b-hf --token YOUR_HF_TOKEN

The model is saved to:   models/<model-slug>/
Re-running is safe — it skips already-downloaded files.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ── recommended models (small → large) ───────────────────────────────────────
RECOMMENDED = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "~2 GB  — fastest, good for testing"),
    ("microsoft/phi-2",                      "~5 GB  — very capable small model"),
    ("mistralai/Mistral-7B-Instruct-v0.2",  "~14 GB — powerful, needs 16 GB+ RAM"),
    ("meta-llama/Llama-2-7b-hf",            "~13 GB — needs HuggingFace token"),
    ("meta-llama/Meta-Llama-3-8B",          "~16 GB — needs HuggingFace token"),
]

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   KV-Cache Compression — Model Downloader                ║
╠══════════════════════════════════════════════════════════╣
║   Recommended models:                                    ║
║     1. TinyLlama/TinyLlama-1.1B-Chat-v1.0  (~2 GB)      ║
║        Best for quick experiments, no token needed       ║
║     2. microsoft/phi-2                      (~5 GB)      ║
║        Excellent quality/size tradeoff                   ║
║     3. mistralai/Mistral-7B-Instruct-v0.2  (~14 GB)     ║
║        Best quality, needs more RAM                      ║
╚══════════════════════════════════════════════════════════╝
"""


def download(model_name: str, save_dir: str, token: str | None, dtype: str) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    slug = model_name.replace("/", "__")
    local_path = os.path.join(save_dir, slug)
    os.makedirs(local_path, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Model  : {model_name}")
    print(f"  Save to: {local_path}")
    print(f"  dtype  : {dtype}")
    print(f"{'─'*60}\n")

    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # ── Download tokenizer ──────────────────────────────────────────────────
    print("⏳ Downloading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(local_path)
    print(f"✅ Tokenizer saved  ({time.time()-t0:.1f}s)\n")

    # ── Download model ──────────────────────────────────────────────────────
    print("⏳ Downloading model weights (this may take a few minutes)...")
    print("   Progress is shown below:\n")
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        token=token,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(local_path)
    elapsed = time.time() - t1

    # ── Size on disk ────────────────────────────────────────────────────────
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(local_path)
        for f in fns
    )
    size_gb = total_bytes / (1024 ** 3)

    print(f"\n✅ Model saved to  : {local_path}")
    print(f"   Download time   : {elapsed:.1f}s")
    print(f"   Size on disk    : {size_gb:.2f} GB")
    print(f"\n{'─'*60}")
    print(f"  To run experiments with this model:")
    print(f"  python experiments/phase5_compare_all.py --model {local_path}")
    print(f"  python experiments/token_visibility.py   --model {local_path}")
    print(f"{'─'*60}\n")


def main() -> None:
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Download a HuggingFace causal LM into models/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--save-dir", default="models",
        help="Root folder to save models"
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace access token (for gated models like LLaMA)"
    )
    parser.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype"
    )
    args = parser.parse_args()

    # Check HF token env var as fallback
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    download(args.model, args.save_dir, token, args.dtype)


if __name__ == "__main__":
    main()
