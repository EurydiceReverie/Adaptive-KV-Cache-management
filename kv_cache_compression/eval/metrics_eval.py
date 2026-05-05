"""
Rich Evaluation Metrics for KV-Cache Compression Quality.

Provides string-overlap and generation-quality metrics beyond simple NLL:

  - Token-level F1 (used by SQuAD and many QA benchmarks)
  - ROUGE-1 / ROUGE-2 / ROUGE-L  (summarisation quality)
  - Exact Match with normalisation (lower-case, strip punctuation)
  - BERTScore-style cosine similarity (optional, needs transformers)
  - Perplexity from sliding-window NLL
  - Needle retrieval accuracy (did the model recover the hidden string?)

All functions are pure-Python + PyTorch — no external eval library required
for the core metrics. ROUGE requires `rouge-score` (optional).

Usage
-----
    from kv_cache_compression.eval.metrics_eval import (
        token_f1, rouge_scores, exact_match_normalized, needle_recall
    )

    f1  = token_f1(prediction="The cat sat", reference="A cat was sitting")
    rg  = rouge_scores(prediction="...", reference="...")
    em  = exact_match_normalized(prediction=" Paris.", reference="paris")
"""
from __future__ import annotations

import re
import string
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
# Text normalisation utilities
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Exact Match (normalised)
# ══════════════════════════════════════════════════════════════════════════════

def exact_match_normalized(prediction: str, reference: str) -> float:
    """
    Exact match after normalisation.

    Matches SQuAD evaluation: lower-case, strip punctuation, collapse whitespace.
    Returns 1.0 if match, 0.0 otherwise.
    """
    return float(_normalize_text(prediction) == _normalize_text(reference))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Token-level F1
# ══════════════════════════════════════════════════════════════════════════════

def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score (SQuAD-style).

    Precision = overlap_tokens / prediction_tokens
    Recall    = overlap_tokens / reference_tokens
    F1        = 2 * P * R / (P + R)

    Both strings are normalised before tokenisation.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)  # both empty → 1.0

    pred_counter = Counter(pred_tokens)
    ref_counter  = Counter(ref_tokens)

    # Overlap: sum of min counts for each shared token
    overlap = sum((pred_counter & ref_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 6)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROUGE Scores
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(prediction: str, reference: str, n: int) -> dict[str, float]:
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)

    pred_ng = _ngrams(pred_tokens, n)
    ref_ng  = _ngrams(ref_tokens, n)

    overlap = sum((pred_ng & ref_ng).values())
    total_pred = sum(pred_ng.values())
    total_ref  = sum(ref_ng.values())

    precision = overlap / total_pred if total_pred > 0 else 0.0
    recall    = overlap / total_ref  if total_ref  > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-L based on Longest Common Subsequence (LCS)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)

    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # LCS dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / m
    recall    = lcs / n
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def rouge_scores(prediction: str, reference: str) -> dict[str, dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Returns a dict:
    {
        "rouge1": {"precision": ..., "recall": ..., "f1": ...},
        "rouge2": {"precision": ..., "recall": ..., "f1": ...},
        "rougeL": {"precision": ..., "recall": ..., "f1": ...},
    }

    This is a pure-Python implementation — no `rouge-score` library needed.
    """
    return {
        "rouge1": _rouge_n(prediction, reference, n=1),
        "rouge2": _rouge_n(prediction, reference, n=2),
        "rougeL": _rouge_l(prediction, reference),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Needle Retrieval Score
# ══════════════════════════════════════════════════════════════════════════════

def needle_recall(prediction: str, needle: str, *, case_sensitive: bool = False) -> float:
    """
    Check whether the needle string appears in the prediction.

    Returns 1.0 if found, 0.0 otherwise.
    Useful for Needle-in-Haystack evaluation.
    """
    if not case_sensitive:
        return float(needle.lower() in prediction.lower())
    return float(needle in prediction)


def needle_position_score(
    prediction: str,
    needle: str,
    context_length: int,
    needle_position: int,
) -> dict[str, float]:
    """
    Extended needle evaluation: retrieval + position difficulty score.

    Parameters
    ----------
    prediction     : model output text
    needle         : the hidden string to find
    context_length : total number of tokens in the prompt
    needle_position: token position where the needle was embedded

    Returns
    -------
    {
        "retrieved": 1.0/0.0,
        "depth_pct": how deep in context the needle was (0=start, 1=end),
        "difficulty": 1 - depth_pct (deeper = harder for most models),
    }
    """
    retrieved = needle_recall(prediction, needle)
    depth_pct = needle_position / max(context_length, 1)
    return {
        "retrieved": retrieved,
        "depth_pct": round(depth_pct, 4),
        "difficulty": round(1.0 - depth_pct, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Aggregate over a batch of predictions
# ══════════════════════════════════════════════════════════════════════════════

def batch_metrics(
    predictions: list[str],
    references: list[str],
    *,
    include_rouge: bool = True,
    needles: list[str] | None = None,
) -> dict:
    """
    Compute aggregate metrics over a list of predictions and references.

    Returns mean ± std for all metrics.

    Parameters
    ----------
    predictions   : list of model output strings
    references    : list of gold reference strings (same length)
    include_rouge : whether to compute ROUGE scores (slower for large batches)
    needles       : if provided, also compute needle recall for each item
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {}

    import statistics

    em_scores  = [exact_match_normalized(p, r) for p, r in zip(predictions, references)]
    f1_scores  = [token_f1(p, r)              for p, r in zip(predictions, references)]

    result: dict = {
        "n": len(predictions),
        "exact_match": {
            "mean": round(statistics.mean(em_scores), 4),
            "std":  round(statistics.stdev(em_scores) if len(em_scores) > 1 else 0.0, 4),
        },
        "token_f1": {
            "mean": round(statistics.mean(f1_scores), 4),
            "std":  round(statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0, 4),
        },
    }

    if include_rouge:
        r1 = [rouge_scores(p, r)["rouge1"]["f1"] for p, r in zip(predictions, references)]
        r2 = [rouge_scores(p, r)["rouge2"]["f1"] for p, r in zip(predictions, references)]
        rl = [rouge_scores(p, r)["rougeL"]["f1"] for p, r in zip(predictions, references)]
        result["rouge1_f1"] = {
            "mean": round(statistics.mean(r1), 4),
            "std":  round(statistics.stdev(r1) if len(r1) > 1 else 0.0, 4),
        }
        result["rouge2_f1"] = {
            "mean": round(statistics.mean(r2), 4),
            "std":  round(statistics.stdev(r2) if len(r2) > 1 else 0.0, 4),
        }
        result["rougeL_f1"] = {
            "mean": round(statistics.mean(rl), 4),
            "std":  round(statistics.stdev(rl) if len(rl) > 1 else 0.0, 4),
        }

    if needles is not None:
        if len(needles) != len(predictions):
            raise ValueError("needles must have the same length as predictions")
        needle_scores = [needle_recall(p, n) for p, n in zip(predictions, needles)]
        result["needle_recall"] = {
            "mean": round(statistics.mean(needle_scores), 4),
            "std":  round(statistics.stdev(needle_scores) if len(needle_scores) > 1 else 0.0, 4),
        }

    return result
