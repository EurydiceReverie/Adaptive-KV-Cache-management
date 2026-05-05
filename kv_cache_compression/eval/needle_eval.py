from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NeedleHaystackSample:
    prompt: str
    continuation: str
    answer: str


def make_needle_haystack_sample(needle: str, repeats: int = 256) -> NeedleHaystackSample:
    filler = " ".join(f"filler_token_{idx}" for idx in range(repeats))
    prompt = (
        "You must remember the hidden key from the long context. "
        f"Context begins. {filler} The hidden key is: {needle}. {filler} Context ends. "
        "Question: What is the hidden key? Answer:"
    )
    continuation = f" {needle}"
    return NeedleHaystackSample(prompt=prompt, continuation=continuation, answer=needle)
