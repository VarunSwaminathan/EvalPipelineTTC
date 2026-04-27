"""CompressingBackend middleware: TTC compress → inner LLM, transparently."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.llm_backend import CompressingBackend, MockBackend


@dataclass
class _FakeCompressionResult:
    output: str
    latency_ms: float = 12.0
    compression_ratio: float = 0.4

class _RecordingTTC:
    """Test double: records every compress() call so assertions are explicit."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def compress(self, text: str, aggressiveness: float = 0.5, model: str | None = None):
        self.calls.append((text, aggressiveness))
        return _FakeCompressionResult(output=f"<compressed:{len(text)}>")

class _FailingTTC:
    """Test double: raises so we can verify the fallback-to-uncompressed path."""

    def compress(self, text: str, aggressiveness: float = 0.5, model: str | None = None):
        raise RuntimeError("simulated TTC outage")

def _long(text: str = "the quick brown fox jumps over the lazy dog. ") -> str:
    """Build a string longer than the default min_chars threshold (200)."""
    return text * 10

def test_answer_compresses_only_the_context() -> None:
    ttc = _RecordingTTC()
    backend = CompressingBackend(MockBackend(), ttc, aggressiveness=0.1)

    long_context = _long()
    backend.answer(long_context, "what is it?")

    assert len(ttc.calls) == 1
    sent_text, sent_aggr = ttc.calls[0]
    assert sent_text == long_context
    assert sent_aggr == 0.1

def test_short_inputs_skip_the_compression_call() -> None:
    ttc = _RecordingTTC()
    backend = CompressingBackend(MockBackend(), ttc, min_chars_to_compress=200)

    backend.answer("Short context.", "Short question?")
    assert ttc.calls == []

def test_summarize_compresses_the_document() -> None:
    ttc = _RecordingTTC()
    backend = CompressingBackend(MockBackend(), ttc, aggressiveness=0.1)

    backend.summarize(_long(), max_words=50)
    assert len(ttc.calls) == 1

def test_chat_compresses_only_long_user_messages() -> None:
    ttc = _RecordingTTC()
    backend = CompressingBackend(MockBackend(), ttc, aggressiveness=0.1)

    messages = [
        {"role": "system", "content": "You are helpful." * 50},
        {"role": "user", "content": _long()},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "thanks"},
    ]
    backend.chat(messages)

    assert len(ttc.calls) == 1

def test_failure_falls_back_to_uncompressed_when_fail_open() -> None:
    """Default fail_open=True: middleware must never make the inner call
    worse than no-compression."""
    backend = CompressingBackend(MockBackend(), _FailingTTC())
    long_context = _long()

    result = backend.answer(long_context, "what?")
    assert isinstance(result, str)
    assert backend.last_compression_ratio == 0.0

def test_failure_propagates_when_fail_open_false() -> None:
    """fail_open=False: TTC failures surface so the caller can decide."""
    backend = CompressingBackend(MockBackend(), _FailingTTC(), fail_open=False)
    with pytest.raises(RuntimeError, match="simulated TTC outage"):
        backend.answer(_long(), "what?")

def test_compress_roles_can_include_system() -> None:
    """Default config skips system prompts; widening compress_roles fires on them."""
    ttc = _RecordingTTC()
    backend = CompressingBackend(
        MockBackend(), ttc,
        compress_roles={"user", "system"},
    )

    long_system = "You are a helpful assistant. " * 30
    backend.chat([
        {"role": "system", "content": long_system},
        {"role": "user", "content": _long()},
    ])
    assert len(ttc.calls) == 2

def test_compress_roles_default_excludes_system_and_assistant() -> None:
    """Sanity check: the default frozenset is exactly {'user'}."""
    ttc = _RecordingTTC()
    backend = CompressingBackend(MockBackend(), ttc)

    backend.chat([
        {"role": "system", "content": _long()},
        {"role": "assistant", "content": _long()},
        {"role": "user", "content": _long()},
    ])
    assert len(ttc.calls) == 1

def test_latency_accounting_includes_both_legs() -> None:
    ttc = _RecordingTTC()
    backend = CompressingBackend(MockBackend(), ttc)

    backend.answer(_long(), "what?")
    assert backend.last_call_latency_ms >= backend.last_compression_latency_ms
    assert backend.last_call_latency_ms >= backend.last_inner_latency_ms
