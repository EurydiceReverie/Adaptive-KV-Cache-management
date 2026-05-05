from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class TimerResult:
    seconds: float = 0.0


@contextmanager
def timed() -> TimerResult:
    start = time.perf_counter()
    result = TimerResult()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"
