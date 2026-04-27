"""Production-style integration of the TTC compression middleware.

Usage:
    # Fill in keys in `.env` (or export them), then:
    python examples/compressed_llm_call.py

What this demonstrates:

* The original LLM call is unchanged at the call site (``llm.answer(...)``).
* The compression step is fully encapsulated in ``CompressingBackend``.
* If TTC is down, the middleware silently falls back to the uncompressed
  prompt — never makes the user's call worse than no-compression. To
  surface the failure instead, pass ``fail_open=False``.
* To compress more than just user-role messages in chat(), pass
  ``compress_roles={"user", "system"}`` (or whatever subset fits).

This is the production analogue of the eval pipeline: same backends, but
compression is woven into each call instead of being a measured separate
step.
"""

from __future__ import annotations

import logging

from eval.client import TTCClient
from eval.llm_backend import CompressingBackend, OpenAIBackend
from eval.logging_config import setup_logging


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)

def main() -> None:
    _load_env()
    setup_logging(level="INFO")
    log = logging.getLogger("example")

    prompt = (
        "In its most fundamental sense, compression is the process of encoding "
        "information using fewer bits than the original representation, in such "
        "a way that the original (or a faithful approximation of it) can be "
        "recovered from the encoded form. The discipline has a long history, "
        "spanning information theory, signal processing, and increasingly, the "
        "preparation of inputs for large language models. The Token Company's "
        "bear-1.x family extends this lineage to the prompt layer: tokens that "
        "carry little information for the downstream task are removed before "
        "the prompt is sent to the model, reducing both cost and latency "
        "without measurable loss of task accuracy in most domains."
    )
    question = "What does compression do?"

    ttc = TTCClient(model="bear-1.2")
    inner = OpenAIBackend(model="gpt-4o-mini")
    llm = CompressingBackend(inner, ttc, aggressiveness=0.1)
    answer = llm.answer(prompt, question)

    log.info(
        "compressed call complete",
        extra={
            "answer": answer[:120],
            "compression_ratio": llm.last_compression_ratio,
            "compression_latency_ms": round(llm.last_compression_latency_ms, 1),
            "llm_latency_ms": round(llm.last_inner_latency_ms, 1),
            "e2e_latency_ms": round(llm.last_call_latency_ms, 1),
        },
    )

if __name__ == "__main__":
    main()
