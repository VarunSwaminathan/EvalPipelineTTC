"""Thin TTC API client.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

TTC_API_BASE = "https://api.thetokencompany.com/v1"
COMPRESS_ENDPOINT = f"{TTC_API_BASE}/compress"

COST_PER_REMOVED_TOKEN_USD = 0.05 / 1_000_000

DEFAULT_MODEL = "bear-1.2"
SUPPORTED_MODELS = ("bear-1", "bear-1.1", "bear-1.2")

DEFAULT_REQUESTS_PER_MINUTE = 50
TTC_DOCUMENTED_LIMIT_PER_MINUTE = 60

@dataclass(frozen=True)
class CompressionResult:
    """A single compression call's full audit trail.
    """

    output: str
    model: str
    aggressiveness: float
    original_tokens: int
    output_tokens: int
    tokens_removed: int
    compression_ratio: float
    latency_ms: float
    ttc_cost_usd: float
    request_id: str
    raw_response: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(
        cls,
        *,
        api_response: dict[str, Any],
        model: str,
        aggressiveness: float,
        latency_ms: float,
        request_id: str,
    ) -> CompressionResult:
        original = int(api_response["original_input_tokens"])
        out = int(api_response["output_tokens"])
        removed = max(0, original - out)
        ratio = removed / original if original > 0 else 0.0
        return cls(
            output=api_response["output"],
            model=model,
            aggressiveness=aggressiveness,
            original_tokens=original,
            output_tokens=out,
            tokens_removed=removed,
            compression_ratio=ratio,
            latency_ms=latency_ms,
            ttc_cost_usd=removed * COST_PER_REMOVED_TOKEN_USD,
            request_id=request_id,
            raw_response=api_response,
        )

class TTCClientError(RuntimeError):
    """Raised when a TTC request fails after all retries."""

class _RateLimiter:
    """Sliding-window rate limiter. Thread-safe.
    """

    def __init__(self, max_per_window: int, window_s: float = 60.0) -> None:
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block (sleep) until the limiter has room for one more call."""
        if self.max_per_window <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                window_start = now - self.window_s
                while self._timestamps and self._timestamps[0] < window_start:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_per_window:
                    self._timestamps.append(now)
                    return
                sleep_for = self.window_s - (now - self._timestamps[0]) + 0.05
            time.sleep(max(0.05, sleep_for))

class TTCClient:
    """Synchronous TTC compression client with retries, gzip, and structured logs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        gzip_enabled: bool = True,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        endpoint: str = COMPRESS_ENDPOINT,
        session: requests.Session | None = None,
        requests_per_minute: int | None = DEFAULT_REQUESTS_PER_MINUTE,
    ) -> None:
        resolved_key = api_key or os.environ.get("TTC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "TTC API key not provided. Pass api_key=<API_KEY> or set TTC_API_KEY env var."
            )
        self._api_key = resolved_key
        self.model = model
        self.gzip_enabled = gzip_enabled
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._endpoint = endpoint
        self._session = session or requests.Session()
        self._rate_limiter: _RateLimiter | None = (
            _RateLimiter(requests_per_minute) if requests_per_minute else None
        )

    def compress(
        self,
        text: str,
        aggressiveness: float = 0.5,
        model: str | None = None,
    ) -> CompressionResult:
        """Compress ``text`` and return the full audit trail.
        """
        if not 0.0 <= aggressiveness <= 1.0:
            raise ValueError(f"aggressiveness must be in [0.0, 1.0]; got {aggressiveness}")

        model_to_use = model or self.model
        request_id = uuid.uuid4().hex[:12]
        payload = {
            "model": model_to_use,
            "input": text,
            "compression_settings": {"aggressiveness": aggressiveness},
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            start = time.perf_counter()
            try:
                response = self._send(payload, request_id=request_id)
                latency_ms = (time.perf_counter() - start) * 1000.0

                if response.status_code >= 500 or response.status_code == 429:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    raise _RetryableHTTPError(
                        response.status_code, response.text, retry_after=retry_after
                    )
                response.raise_for_status()

                data = response.json()
                result = CompressionResult.from_api(
                    api_response=data,
                    model=model_to_use,
                    aggressiveness=aggressiveness,
                    latency_ms=latency_ms,
                    request_id=request_id,
                )
                logger.info(
                    "ttc compress ok",
                    extra={
                        "request_id": request_id,
                        "model": result.model,
                        "aggressiveness": result.aggressiveness,
                        "input_tokens": result.original_tokens,
                        "output_tokens": result.output_tokens,
                        "removed_tokens": result.tokens_removed,
                        "compression_ratio": round(result.compression_ratio, 4),
                        "latency_ms": round(result.latency_ms, 2),
                        "cost_usd": round(result.ttc_cost_usd, 8),
                        "attempt": attempt,
                    },
                )
                return result

            except (_RetryableHTTPError, requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "ttc compress retryable error",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error": str(exc),
                    },
                )
                if attempt >= self.max_retries:
                    break
                retry_after = getattr(last_exc, "retry_after", None)
                self._sleep_backoff(attempt, retry_after=retry_after)

            except requests.HTTPError as exc:
                logger.error(
                    "ttc compress http error (non-retryable)",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "status_code": exc.response.status_code if exc.response else None,
                        "error": str(exc),
                    },
                )
                raise TTCClientError(f"TTC HTTP error: {exc}") from exc

        raise TTCClientError(
            f"TTC compress failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    def _send(self, payload: dict[str, Any], *, request_id: str) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        }
        body = json.dumps(payload).encode("utf-8")
        if self.gzip_enabled:
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"
        return self._session.post(
            self._endpoint,
            data=body,
            headers=headers,
            timeout=self.timeout_s,
        )

    def _sleep_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        if retry_after is not None and retry_after > 0:
            time.sleep(min(retry_after, 30.0))
            return
        base = min(2.0 ** (attempt - 1), 8.0)
        time.sleep(random.uniform(0.0, base))

class _RetryableHTTPError(Exception):
    """Internal sentinel for 5xx + 429 responses. Not part of the public API."""

    def __init__(self, status_code: int, body: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after

def _parse_retry_after(value: str | None) -> float | None:
    """Parse the ``Retry-After`` header. Spec allows seconds-int or HTTP-date.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None

class MockTTCClient:
    """Drop-in replacement for ``TTCClient`` that works without API keys.
    """

    _LOW_SIGNAL_TOKENS = frozenset({
        "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "with",
        "is", "are", "was", "were", "be", "been", "being", "by", "as", "at",
        "this", "that", "these", "those", "it", "its", "from", "but",
        "very", "quite", "rather", "indeed", "actually", "essentially",
        "fundamentally", "basically", "primarily", "particularly", "notably",
        "specifically", "generally", "typically", "usually", "often",
        "sometimes", "however", "moreover", "furthermore", "additionally",
        "consequently", "subsequently", "previously", "nevertheless",
        "approximately", "roughly", "modestly", "substantially",
    })

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.gzip_enabled = False
        self.max_retries = 0

    def compress(
        self,
        text: str,
        aggressiveness: float = 0.5,
        model: str | None = None,
    ) -> CompressionResult:
        if not 0.0 <= aggressiveness <= 1.0:
            raise ValueError(f"aggressiveness must be in [0.0, 1.0]; got {aggressiveness}")

        protected_segments: list[str] = []
        scrubbed = text

        def _stash(match):
            protected_segments.append(match.group(0))
            return f"\x00TTC_SAFE_{len(protected_segments) - 1}\x00"

        import re
        scrubbed = re.sub(
            r"<ttc_safe>.*?</ttc_safe>",
            _stash,
            scrubbed,
            flags=re.DOTALL,
        )

        original_tokens = max(1, len(scrubbed.split()))
        words = scrubbed.split()
        kept: list[str] = []
        for w in words:
            stripped = w.strip(",.;:!?\"'()[]{}—-").lower()
            if stripped in self._LOW_SIGNAL_TOKENS:
                import hashlib
                bucket = int(hashlib.md5(f"{stripped}:{aggressiveness}".encode()).hexdigest()[:8], 16) % 1000
                if bucket < int(aggressiveness * 1000):
                    continue
            kept.append(w)

        compressed = " ".join(kept)
        for i, segment in enumerate(protected_segments):
            compressed = compressed.replace(f"\x00TTC_SAFE_{i}\x00", segment)

        output_tokens = max(1, len(compressed.split()))
        api_response = {
            "output": compressed,
            "original_input_tokens": original_tokens,
            "output_tokens": output_tokens,
        }
        latency_ms = 1.0 + len(text) / 10_000.0
        return CompressionResult.from_api(
            api_response=api_response,
            model=model or self.model,
            aggressiveness=aggressiveness,
            latency_ms=latency_ms,
            request_id=uuid.uuid4().hex[:12],
        )
