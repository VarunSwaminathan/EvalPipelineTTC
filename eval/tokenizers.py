"""Downstream-LLM token counting and cost estimation.

The TTC bill is one number; the *actual* customer savings come from sending
fewer tokens to the downstream LLM. The downstream count depends on the
tokenizer the downstream model uses (OpenAI's BPE here), which in general
is *not* the same tokenizer TTC uses internally. So we count downstream
tokens with the downstream tokenizer.

Two resolution paths:

1. ``tiktoken`` for OpenAI / GPT-4o / GPT-4o-mini (the only supported backend).
2. Whitespace-split heuristic fallback (logged as a warning) so the smoke
   test runs without ``tiktoken`` installed.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

PRICE_TABLE_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

_OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}

_warned_fallback = False

def count_tokens(text: str, model: str) -> int:
    """Tokenize ``text`` for the given downstream model and return the count."""
    if not text:
        return 0
    if _is_openai(model):
        return _count_tiktoken(text, model)
    return _count_fallback(text)

def estimate_llm_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Dollars for one LLM call given input + output token counts."""
    prices = PRICE_TABLE_USD_PER_1M.get(model)
    if prices is None:
        logger.warning(
            "no price entry for model; cost estimate will be 0",
            extra={"model": model},
        )
        return 0.0
    return (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )

def _is_openai(model: str) -> bool:
    return model in _OPENAI_MODELS or model.startswith("gpt-")

@lru_cache(maxsize=8)
def _tiktoken_encoder(model: str):  # type: ignore[no-untyped-def]
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")

def _count_tiktoken(text: str, model: str) -> int:
    try:
        enc = _tiktoken_encoder(model)
        return len(enc.encode(text))
    except ImportError:
        return _count_fallback(text)

def _count_fallback(text: str) -> int:
    """Whitespace-split heuristic. Wildly imprecise for non-Latin scripts."""
    global _warned_fallback
    if not _warned_fallback:
        logger.warning(
            "falling back to whitespace token counter — install tiktoken for accuracy"
        )
        _warned_fallback = True
    return max(1, len(text.split()))
