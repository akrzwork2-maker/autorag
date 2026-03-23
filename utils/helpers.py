from __future__ import annotations

import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        return result, elapsed
    return wrapper


def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def confidence_color(c_f: float) -> str:
    if c_f >= 0.70:
        return "green"
    elif c_f >= 0.50:
        return "orange"
    return "red"


def verdict_emoji(verdict: str) -> str:
    return {
        "VERIFIED": "🟢",
        "CONTRADICTED": "🔴",
        "UNVERIFIABLE": "🟡",
    }.get(verdict, "⚪")


def format_score(score: float, decimals: int = 3) -> str:
    return f"{score:.{decimals}f}"
