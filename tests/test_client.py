"""TTCClient: gzip-by-default, retry on 5xx, no retry on 4xx, cost math."""

from __future__ import annotations

import gzip
import json
import time

import pytest
import responses

from eval.client import COMPRESS_ENDPOINT, TTCClient, TTCClientError, _RateLimiter


def _ok_body(input_tokens: int = 100, output_tokens: int = 60, output: str = "x") -> dict:
    return {
        "output": output,
        "original_input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

@responses.activate
def test_gzip_header_set_by_default() -> None:
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=_ok_body(), status=200)
    client = TTCClient(api_key="fake-key")
    client.compress("hello world", aggressiveness=0.5)

    sent = responses.calls[0].request
    assert sent.headers.get("Content-Encoding") == "gzip"
    decoded = json.loads(gzip.decompress(sent.body).decode("utf-8"))
    assert decoded["model"] == "bear-1.2"
    assert decoded["input"] == "hello world"
    assert decoded["compression_settings"] == {"aggressiveness": 0.5}

@responses.activate
def test_gzip_disabled_sends_plain_json() -> None:
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=_ok_body(), status=200)
    client = TTCClient(api_key="fake-key", gzip_enabled=False)
    client.compress("hello world")

    sent = responses.calls[0].request
    assert sent.headers.get("Content-Encoding") is None
    decoded = json.loads(sent.body.decode("utf-8"))
    assert decoded["input"] == "hello world"

@responses.activate
def test_retry_on_500_then_success() -> None:
    responses.add(responses.POST, COMPRESS_ENDPOINT, status=500, body="upstream is sad")
    responses.add(responses.POST, COMPRESS_ENDPOINT, status=503, body="still sad")
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=_ok_body(), status=200)

    client = TTCClient(api_key="fake-key", max_retries=3)
    client._sleep_backoff = lambda attempt, retry_after=None: None  # type: ignore[method-assign]

    result = client.compress("hello world")
    assert result.output == "x"
    assert len(responses.calls) == 3

@responses.activate
def test_no_retry_on_400() -> None:
    responses.add(responses.POST, COMPRESS_ENDPOINT, status=400, body="bad input")

    client = TTCClient(api_key="fake-key", max_retries=3)
    client._sleep_backoff = lambda attempt, retry_after=None: None  # type: ignore[method-assign]

    with pytest.raises(TTCClientError):
        client.compress("hello world")
    assert len(responses.calls) == 1

@responses.activate
def test_retry_on_429_rate_limit() -> None:
    """429 is the server saying 'slow down' — that IS retryable.

    Regression test: an earlier version of the client raised on every 4xx
    including 429, which caused real eval runs to fail two whole tasks
    when TTC's rate limit kicked in mid-sweep.
    """
    responses.add(responses.POST, COMPRESS_ENDPOINT, status=429, body="rate limited")
    responses.add(responses.POST, COMPRESS_ENDPOINT, status=429, body="still rate limited")
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=_ok_body(), status=200)

    client = TTCClient(api_key="fake-key", max_retries=3)
    client._sleep_backoff = lambda attempt, retry_after=None: None  # type: ignore[method-assign]

    result = client.compress("hello world")
    assert result.output == "x"
    assert len(responses.calls) == 3

@responses.activate
def test_retry_after_header_is_honored() -> None:
    """When the server sends Retry-After, the client should sleep for that long."""
    responses.add(
        responses.POST, COMPRESS_ENDPOINT,
        status=429, body="rate limited",
        headers={"Retry-After": "2"},
    )
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=_ok_body(), status=200)

    client = TTCClient(api_key="fake-key", max_retries=3)
    sleeps: list[float | None] = []
    client._sleep_backoff = (  # type: ignore[method-assign]
        lambda attempt, retry_after=None: sleeps.append(retry_after)
    )

    client.compress("hello world")
    assert sleeps == [2.0]

@responses.activate
def test_cost_calculation_uses_removed_tokens() -> None:
    responses.add(
        responses.POST,
        COMPRESS_ENDPOINT,
        json=_ok_body(input_tokens=2_000_000, output_tokens=1_000_000),
        status=200,
    )
    client = TTCClient(api_key="fake-key")
    result = client.compress("any text")

    assert result.tokens_removed == 1_000_000
    assert result.ttc_cost_usd == pytest.approx(0.05)
    assert result.compression_ratio == pytest.approx(0.5)

@responses.activate
def test_aggressiveness_validated() -> None:
    client = TTCClient(api_key="fake-key")
    with pytest.raises(ValueError):
        client.compress("x", aggressiveness=1.5)

def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TTC API key"):
        TTCClient()

def test_env_var_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTC_API_KEY", "from-env")
    client = TTCClient()
    assert client._api_key == "from-env"

def test_rate_limiter_allows_burst_up_to_cap() -> None:
    """First N calls within the window are non-blocking."""
    rl = _RateLimiter(max_per_window=3, window_s=1.0)
    start = time.monotonic()
    for _ in range(3):
        rl.acquire()
    assert time.monotonic() - start < 0.1

def test_rate_limiter_blocks_when_full() -> None:
    """The (cap+1)th call must wait for the window to slide.

    Uses a 1-second window so the test completes quickly. Key invariant:
    after acquiring 2 immediately, the 3rd acquire takes ~1s of wall time.
    """
    rl = _RateLimiter(max_per_window=2, window_s=1.0)
    rl.acquire()
    rl.acquire()
    start = time.monotonic()
    rl.acquire()
    elapsed = time.monotonic() - start
    assert 0.7 < elapsed < 1.5

def test_rate_limiter_disabled_when_zero() -> None:
    """``max_per_window=0`` is the documented disable value — must not block."""
    rl = _RateLimiter(max_per_window=0, window_s=60.0)
    start = time.monotonic()
    for _ in range(100):
        rl.acquire()
    assert time.monotonic() - start < 0.05

@responses.activate
def test_client_paces_under_rate_limit() -> None:
    """End-to-end: TTCClient with rpm=2 forces the 3rd compress() to wait.

    This is the regression test for the actual bug we hit: bursts of 5
    compressions in quick succession (RAG fan-out) were tripping TTC's 60/min
    server limit. With pacing enabled, the (cap+1)th call waits.
    """
    for _ in range(3):
        responses.add(responses.POST, COMPRESS_ENDPOINT, json=_ok_body(), status=200)

    client = TTCClient(api_key="fake-key", requests_per_minute=2)
    client._rate_limiter = _RateLimiter(max_per_window=2, window_s=0.5)

    start = time.monotonic()
    client.compress("a")
    client.compress("b")
    client.compress("c")
    elapsed = time.monotonic() - start
    assert elapsed > 0.35
    assert len(responses.calls) == 3

def test_client_pacing_can_be_disabled() -> None:
    """Pass requests_per_minute=None to opt out (e.g., higher-tier quota)."""
    client = TTCClient(api_key="fake-key", requests_per_minute=None)
    assert client._rate_limiter is None
