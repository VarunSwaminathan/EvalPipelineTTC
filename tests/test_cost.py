"""End-to-end cost math: TTC bill + downstream LLM bill must match expectations."""

from __future__ import annotations

import pytest

from eval.client import COST_PER_REMOVED_TOKEN_USD
from eval.tokenizers import PRICE_TABLE_USD_PER_1M, estimate_llm_cost_usd


def test_estimate_gpt_4o_mini_known_price() -> None:
    assert estimate_llm_cost_usd(1_000_000, 1_000_000, "gpt-4o-mini") == pytest.approx(0.75)

def test_estimate_unknown_model_returns_zero() -> None:
    assert estimate_llm_cost_usd(1000, 1000, "completely-made-up-model") == 0.0

def test_net_savings_worked_example() -> None:
    """End-to-end customer ROI math.

    Scenario: 10K input tokens of context, compressed to 6K (40% removed),
    sent to gpt-4o-mini for a 200-token response. We compare the net cost
    against the LLM-only baseline.
    """
    original_input, compressed_input, output = 10_000, 6_000, 200
    removed = original_input - compressed_input

    ttc_cost = removed * COST_PER_REMOVED_TOKEN_USD
    llm_cost_baseline = estimate_llm_cost_usd(original_input, output, "gpt-4o-mini")
    llm_cost_compressed = estimate_llm_cost_usd(compressed_input, output, "gpt-4o-mini")

    net_savings = (llm_cost_baseline - llm_cost_compressed) - ttc_cost

    assert net_savings > 0
    assert ttc_cost < (llm_cost_baseline - llm_cost_compressed)

def test_price_table_has_all_required_models() -> None:
    required = {"gpt-4o-mini", "gpt-4o"}
    assert required.issubset(PRICE_TABLE_USD_PER_1M.keys())
