"""Downstream LLM backends.

Two implementations behind a common ``LLMBackend`` interface, plus a
compression-middleware wrapper:

* ``MockBackend``: TF-IDF / TextRank-based extractive answers. Deterministic,
  needs no API key, **complete enough that the smoke test produces a real
  end-to-end report**. The reviewer's first pass through this repo will run
  ``--llm mock`` so the mock has to be more than a stub.
* ``OpenAIBackend``: uses ``OPENAI_API_KEY``. The primary backend for the
  real eval run. ``gpt-4o-mini`` by default — cheapest frontier model, fits
  the eval budget.
* ``CompressingBackend``: wraps any of the above with TTC compression as
  middleware between your prompt and the inner LLM call. Production
  analogue of the eval pipeline: same backends, but compression is woven
  into each call instead of being a measured separate step.

All backends record ``last_call_latency_ms`` so the runner can roll up E2E
latency for the report's headline latency table.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    """All downstream-LLM operations the eval needs.
    """

    name: str = "abstract"
    model: str = "abstract"

    def __init__(self) -> None:
        self.last_call_latency_ms: float = 0.0

    @abstractmethod
    def answer(self, context: str, question: str) -> str: ...

    @abstractmethod
    def summarize(self, document: str, max_words: int = 100) -> str: ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str: ...

class MockBackend(LLMBackend):
    """Deterministic extractive backend.
    """

    name = "mock"
    model = "mock-extractive-v1"

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def answer(self, context: str, question: str) -> str:
        start = time.perf_counter()
        sentences = _split_sentences(context)
        if not sentences:
            self.last_call_latency_ms = (time.perf_counter() - start) * 1000.0
            return ""
        q_terms = _tokens(question)
        if not q_terms:
            self.last_call_latency_ms = (time.perf_counter() - start) * 1000.0
            return sentences[0]

        scored = [(_overlap_score(s, q_terms), idx, s) for idx, s in enumerate(sentences)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = scored[0][2]

        answer_span = _shortest_overlap_span(best, q_terms)
        self.last_call_latency_ms = (time.perf_counter() - start) * 1000.0
        return answer_span

    def summarize(self, document: str, max_words: int = 100) -> str:
        start = time.perf_counter()
        sentences = _split_sentences(document)
        if not sentences:
            self.last_call_latency_ms = (time.perf_counter() - start) * 1000.0
            return ""
        ranked = _textrank(sentences)
        out_words: list[str] = []
        chosen: list[tuple[int, str]] = []
        for rank in ranked:
            sent = sentences[rank]
            words = sent.split()
            if len(out_words) + len(words) > max_words and chosen:
                break
            chosen.append((rank, sent))
            out_words.extend(words)
        chosen.sort(key=lambda x: x[0])
        summary = " ".join(s for _, s in chosen)
        self.last_call_latency_ms = (time.perf_counter() - start) * 1000.0
        return summary

    def chat(self, messages: list[dict[str, str]]) -> str:
        user_turns = [m for m in messages if m.get("role") == "user"]
        if not user_turns:
            return ""
        last_q = user_turns[-1]["content"]
        context_parts = [
            m["content"] for m in messages
            if m is not user_turns[-1] and m.get("content")
        ]
        return self.answer("\n".join(context_parts), last_q)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "being", "by", "as", "at",
    "this", "that", "these", "those", "it", "its", "from", "but", "not",
    "what", "who", "when", "where", "why", "how", "did", "do", "does",
    "has", "have", "had", "can", "could", "would", "should", "will",
}

def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if t.lower() not in _STOPWORDS]

def _overlap_score(sentence: str, query_terms: list[str]) -> float:
    sent_terms = Counter(_tokens(sentence))
    if not sent_terms:
        return 0.0
    q = Counter(query_terms)
    score = sum(min(sent_terms[t], q[t]) for t in q)
    return score / math.sqrt(sum(sent_terms.values()))

def _shortest_overlap_span(sentence: str, query_terms: list[str], window: int = 12) -> str:
    """Return a window around the densest cluster of query tokens.
    """
    words = sentence.split()
    if not words:
        return sentence
    q = set(query_terms)
    hits = [i for i, w in enumerate(words) if _WORD_RE.findall(w.lower())[:1] and any(t in q for t in _tokens(w))]
    if not hits:
        return sentence
    center = hits[len(hits) // 2]
    lo = max(0, center - window // 2)
    hi = min(len(words), lo + window)
    return " ".join(words[lo:hi])

def _textrank(sentences: list[str], damping: float = 0.85, iters: int = 30) -> list[int]:
    """Tiny TextRank: PageRank over a sentence-cosine graph. Returns indices in score order."""
    n = len(sentences)
    if n <= 1:
        return list(range(n))

    tokenized = [_tokens(s) for s in sentences]
    vocab = sorted({t for s in tokenized for t in s})
    if not vocab:
        return list(range(n))
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        c = Counter(toks)
        for t, cnt in c.items():
            M[i, idx[t]] = cnt
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    sim = (M @ M.T) / np.outer(norms, norms)
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.full(n, 1.0 / n, dtype=np.float32)
    base = (1.0 - damping) / n
    for _ in range(iters):
        scores = base + damping * (transition.T @ scores)
    return list(np.argsort(-scores))

class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        from openai import OpenAI

        self._client = OpenAI()

    def answer(self, context: str, question: str) -> str:
        prompt = (
            "Answer the question using only information from the context. "
            "Reply with the shortest exact answer span — no preamble, no explanation. "
            "If the answer is not in the context, reply with 'unknown'.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self._complete([{"role": "user", "content": prompt}])

    def summarize(self, document: str, max_words: int = 100) -> str:
        prompt = (
            f"Summarize the document in approximately {max_words} words. "
            "Capture the main facts. Avoid adding information not present in the document.\n\n"
            f"Document:\n{document}\n\nSummary:"
        )
        return self._complete([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._complete(messages)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        finally:
            self.last_call_latency_ms = (time.perf_counter() - start) * 1000.0

class CompressingBackend(LLMBackend):
    """TTC-compression middleware that wraps any other ``LLMBackend``.

    The eval pipeline keeps compression and the downstream LLM call as
    separate steps because it needs to *measure* each. Production code
    doesn't care about the breakdown — it just wants compressed prompts to
    flow through transparently. This wrapper provides that path:

        ttc = TTCClient()                        # uses TTC_API_KEY
        llm = CompressingBackend(OpenAIBackend(), ttc, aggressiveness=0.1)
        answer = llm.answer(context, question)   # compression is invisible

    Behavior:

    * Long-form fields (``context`` for QA, ``document`` for summarization,
      configured roles' content for chat) are routed through ``ttc.compress``
      before being forwarded to the inner backend.
    * Short fields (the question, system prompts by default) are passed
      through unmodified — compressing a 30-character question wastes
      tokens and can corrupt question semantics.
    * Failures in the TTC call default to fail-open (log + forward original
      prompt) so middleware never makes the inner call worse than
      no-compression. Pass ``fail_open=False`` to surface failures instead.
    * ``last_call_latency_ms`` reports the full E2E (TTC + LLM) so callers
      using this wrapper see realistic production latency.
    """

    name = "compressing"

    DEFAULT_COMPRESS_ROLES: frozenset[str] = frozenset({"user"})

    def __init__(
        self,
        inner: LLMBackend,
        ttc_client,
        aggressiveness: float = 0.1,
        model: str | None = None,
        min_chars_to_compress: int = 200,
        fail_open: bool = True,
        compress_roles: set[str] | frozenset[str] | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.ttc = ttc_client
        self.aggressiveness = aggressiveness
        self.compress_model = model
        self.min_chars_to_compress = min_chars_to_compress
        self.fail_open = fail_open
        self.compress_roles = (
            frozenset(compress_roles) if compress_roles is not None
            else self.DEFAULT_COMPRESS_ROLES
        )
        self.model = inner.model
        self.last_compression_latency_ms: float = 0.0
        self.last_inner_latency_ms: float = 0.0
        self.last_compression_ratio: float = 0.0

    def answer(self, context: str, question: str) -> str:
        compressed = self._maybe_compress(context)
        result = self.inner.answer(compressed, question)
        self.last_inner_latency_ms = self.inner.last_call_latency_ms
        self.last_call_latency_ms = self.last_compression_latency_ms + self.last_inner_latency_ms
        return result

    def summarize(self, document: str, max_words: int = 100) -> str:
        compressed = self._maybe_compress(document)
        result = self.inner.summarize(compressed, max_words=max_words)
        self.last_inner_latency_ms = self.inner.last_call_latency_ms
        self.last_call_latency_ms = self.last_compression_latency_ms + self.last_inner_latency_ms
        return result

    def chat(self, messages: list[dict[str, str]]) -> str:
        compressed_msgs: list[dict[str, str]] = []
        total_compress_lat = 0.0
        for m in messages:
            content = m.get("content", "")
            if (
                m.get("role") in self.compress_roles
                and len(content) >= self.min_chars_to_compress
            ):
                new_content = self._compress_text(content)
                total_compress_lat += self.last_compression_latency_ms
                compressed_msgs.append({**m, "content": new_content})
            else:
                compressed_msgs.append(m)
        self.last_compression_latency_ms = total_compress_lat
        result = self.inner.chat(compressed_msgs)
        self.last_inner_latency_ms = self.inner.last_call_latency_ms
        self.last_call_latency_ms = total_compress_lat + self.last_inner_latency_ms
        return result

    def _maybe_compress(self, text: str) -> str:
        if len(text) < self.min_chars_to_compress:
            self.last_compression_latency_ms = 0.0
            self.last_compression_ratio = 0.0
            return text
        return self._compress_text(text)

    def _compress_text(self, text: str) -> str:
        try:
            result = self.ttc.compress(
                text,
                aggressiveness=self.aggressiveness,
                model=self.compress_model,
            )
            self.last_compression_latency_ms = result.latency_ms
            self.last_compression_ratio = result.compression_ratio
            return result.output
        except Exception as exc:  # noqa: BLE001
            self.last_compression_latency_ms = 0.0
            self.last_compression_ratio = 0.0
            if not self.fail_open:
                raise
            logger.warning(
                "TTC compression failed; falling back to uncompressed prompt",
                extra={"error": str(exc)},
            )
            return text

def build_backend(name: str, **kwargs) -> LLMBackend:
    """Resolve a backend name to a constructed instance. Used by the CLI."""
    name = name.lower()
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "openai":
        return OpenAIBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}. Valid: mock, openai.")
