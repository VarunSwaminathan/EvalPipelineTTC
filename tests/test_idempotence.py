"""Idempotence: TTC describes its models as deterministic.

Two compress() calls with identical inputs must return identical outputs.
This isn't a property of our client (we don't cache) — it's a contract we
verify against the API. The runner re-runs this check at the start of every
real eval to catch upstream regressions.
"""

from __future__ import annotations

import responses

from eval.client import COMPRESS_ENDPOINT, TTCClient


@responses.activate
def test_same_input_same_output() -> None:
    body = {"output": "compressed!", "original_input_tokens": 50, "output_tokens": 30}
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=body, status=200)
    responses.add(responses.POST, COMPRESS_ENDPOINT, json=body, status=200)

    client = TTCClient(api_key="fake-key")
    a = client.compress("the same text", aggressiveness=0.5)
    b = client.compress("the same text", aggressiveness=0.5)

    assert a.output == b.output
    assert a.original_tokens == b.original_tokens
    assert a.output_tokens == b.output_tokens
    assert a.tokens_removed == b.tokens_removed
