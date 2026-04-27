"""Shared helpers used by every task.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from eval.metrics import bootstrap_ci

from .base import SamplePrediction, SweepCell

logger = logging.getLogger(__name__)

def aggregate_with_ci(
    metric_name: str,
    predictions: Iterable[SamplePrediction],
) -> dict[str, float]:
    values = [
        p.metrics[metric_name]
        for p in predictions
        if not p.failed and metric_name in p.metrics
    ]
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = sum(values) / len(values)
    lo, hi = bootstrap_ci(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)}

def build_sweep_cell(
    model: str,
    aggressiveness: float,
    metric_names: list[str],
    predictions: list[SamplePrediction],
    total_samples: int,
) -> SweepCell:
    cell = SweepCell(model=model, aggressiveness=aggressiveness)
    cell.sample_predictions = predictions

    successful = [p for p in predictions if not p.failed]
    cell.failure_count = len(predictions) - len(successful)
    cell.failure_rate = cell.failure_count / total_samples if total_samples else 0.0

    if not successful:
        return cell

    for m in metric_names:
        cell.metrics[m] = aggregate_with_ci(m, successful)

    cell.avg_compression_ratio = _mean(p.compression_ratio for p in successful)
    cell.avg_ttc_cost_usd = _mean(p.ttc_cost_usd for p in successful)

    ttc_lats = [p.ttc_latency_ms for p in successful]
    cell.avg_ttc_latency_ms = _mean(ttc_lats)
    cell.p50_ttc_latency_ms = _percentile(ttc_lats, 50)
    cell.p95_ttc_latency_ms = _percentile(ttc_lats, 95)
    cell.p99_ttc_latency_ms = _percentile(ttc_lats, 99)

    llm_lats = [p.llm_latency_ms for p in successful]
    cell.avg_llm_latency_ms = _mean(llm_lats)
    cell.p50_llm_latency_ms = _percentile(llm_lats, 50)
    cell.p95_llm_latency_ms = _percentile(llm_lats, 95)
    cell.p99_llm_latency_ms = _percentile(llm_lats, 99)

    e2e_lats = [p.e2e_latency_ms for p in successful]
    cell.avg_e2e_latency_ms = _mean(e2e_lats)
    cell.p50_e2e_latency_ms = _percentile(e2e_lats, 50)
    cell.p95_e2e_latency_ms = _percentile(e2e_lats, 95)
    cell.p99_e2e_latency_ms = _percentile(e2e_lats, 99)

    cell.avg_llm_input_tokens = _mean(p.llm_input_tokens for p in successful)
    cell.avg_llm_output_tokens = _mean(p.llm_output_tokens for p in successful)

    return cell

def select_best_cell(
    cells: list[SweepCell],
    primary_metric: str,
    baseline_score: float,
    lambda_compression: float = 0.5,
) -> SweepCell | None:
    """Pareto-knee selector: ``score = accuracy_delta + λ * compression_ratio``.
    """
    if not cells:
        return None

    candidates: list[tuple[float, SweepCell]] = []
    for cell in cells:
        if primary_metric not in cell.metrics:
            continue
        cell_score = cell.metrics[primary_metric]["mean"]
        delta = cell_score - baseline_score
        if delta < -0.02:
            continue
        utility = delta + lambda_compression * cell.avg_compression_ratio
        candidates.append((utility, cell))

    if not candidates:
        cells_with_metric = [c for c in cells if primary_metric in c.metrics]
        if not cells_with_metric:
            return cells[0]
        return max(cells_with_metric, key=lambda c: c.metrics[primary_metric]["mean"])

    return max(candidates, key=lambda x: x[0])[1]

def parallel_map(
    func, items: list, max_workers: int
) -> list:
    """Tiny ThreadPoolExecutor wrapper that preserves input order.
    """
    if max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(func, items))

def _mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (pct / 100.0) * (len(sorted_vals) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

__all__ = [
    "aggregate_with_ci",
    "build_sweep_cell",
    "select_best_cell",
    "parallel_map",
    "statistics",
]
