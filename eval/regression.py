"""Regression detection — compare a current eval run against a prior baseline.

The intended use case is internal: ship ``bear-1.3`` without surprising
customers. You run the eval against the new model, point ``--baseline`` at
the JSON from the last accepted run, and the report's regression section
flags any (task, model, aggressiveness) cell whose accuracy moved by more
than the threshold.

We diff at the metric level — accuracy means (EM/F1/ROUGE-L/retrieval),
compression ratio, faithfulness, p95 E2E latency — because that's what
actually breaks customer experience. We don't diff per-sample predictions:
those are noisy and the report would drown in spurious flags.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CellDiff:
    task: str
    model: str
    aggressiveness: float
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    delta_pct: float | None
    is_regression: bool
    is_improvement: bool

@dataclass
class RegressionReport:
    threshold_pp: float
    threshold_pct: float
    regressions: list[CellDiff] = field(default_factory=list)
    improvements: list[CellDiff] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

LATENCY_METRICS = {"p95_e2e_latency_ms", "p95_llm_latency_ms", "p95_ttc_latency_ms"}

def compare_runs(
    current: dict[str, Any],
    baseline: dict[str, Any],
    threshold_pp: float = 1.0,
    threshold_pct: float = 5.0,
) -> RegressionReport:
    """Compare two saved eval reports (the JSON dicts produced by the runner).

    Args:
        current: dict from the most-recent run (the EvalReport JSON).
        baseline: dict from the prior accepted run.
        threshold_pp: percentage-point threshold for accuracy-style metrics in [0, 1].
        threshold_pct: percent threshold for ratio metrics (compression, latency).
    """
    report = RegressionReport(threshold_pp=threshold_pp, threshold_pct=threshold_pct)

    base_tasks = {t["task_name"]: t for t in baseline.get("tasks", [])}
    cur_tasks = {t["task_name"]: t for t in current.get("tasks", [])}

    for task_name, cur_task in cur_tasks.items():
        base_task = base_tasks.get(task_name)
        if base_task is None:
            continue
        for cur_cell, base_cell in _matched_cells(cur_task, base_task):
            for metric_name, cur_val in _iter_metric_values(cur_cell):
                base_val = _lookup_metric_value(base_cell, metric_name)
                if base_val is None:
                    continue
                diff = _evaluate_diff(
                    task=task_name,
                    cell=cur_cell,
                    metric_name=metric_name,
                    baseline_value=base_val,
                    current_value=cur_val,
                    threshold_pp=threshold_pp,
                    threshold_pct=threshold_pct,
                )
                if diff is None:
                    continue
                if diff.is_regression:
                    report.regressions.append(diff)
                elif diff.is_improvement:
                    report.improvements.append(diff)

    report.regressions.sort(key=lambda d: d.delta if d.metric in LATENCY_METRICS else -abs(d.delta))
    report.improvements.sort(
        key=lambda d: -d.delta if d.metric in LATENCY_METRICS else -abs(d.delta)
    )
    report.summary = {
        "regressions_found": len(report.regressions),
        "improvements_found": len(report.improvements),
        "tasks_compared": len(set(cur_tasks) & set(base_tasks)),
    }
    return report

def load_report_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def _matched_cells(cur_task: dict, base_task: dict):  # type: ignore[no-untyped-def]
    """Yield (current_cell, baseline_cell) pairs matched by (model, aggressiveness)."""
    base_idx = {
        (c["model"], round(c["aggressiveness"], 4)): c
        for c in base_task.get("sweep_results", [])
    }
    for cur_cell in cur_task.get("sweep_results", []):
        key = (cur_cell["model"], round(cur_cell["aggressiveness"], 4))
        base_cell = base_idx.get(key)
        if base_cell is not None:
            yield cur_cell, base_cell

def _iter_metric_values(cell: dict):  # type: ignore[no-untyped-def]
    """Yield (metric_name, value) pairs for everything we want to diff."""
    for metric_name, stats in cell.get("metrics", {}).items():
        yield metric_name, stats.get("mean", 0.0)
    yield "compression_ratio", cell.get("avg_compression_ratio", 0.0)
    yield "p95_e2e_latency_ms", cell.get("p95_e2e_latency_ms", 0.0)
    yield "p95_llm_latency_ms", cell.get("p95_llm_latency_ms", 0.0)
    yield "p95_ttc_latency_ms", cell.get("p95_ttc_latency_ms", 0.0)

def _lookup_metric_value(cell: dict, metric_name: str) -> float | None:  # type: ignore[no-untyped-def]
    if metric_name in cell.get("metrics", {}):
        return float(cell["metrics"][metric_name].get("mean", 0.0))
    direct = {
        "compression_ratio": cell.get("avg_compression_ratio"),
        "p95_e2e_latency_ms": cell.get("p95_e2e_latency_ms"),
        "p95_llm_latency_ms": cell.get("p95_llm_latency_ms"),
        "p95_ttc_latency_ms": cell.get("p95_ttc_latency_ms"),
    }
    val = direct.get(metric_name)
    return float(val) if val is not None else None

def _evaluate_diff(
    *,
    task: str,
    cell: dict[str, Any],
    metric_name: str,
    baseline_value: float,
    current_value: float,
    threshold_pp: float,
    threshold_pct: float,
) -> CellDiff | None:
    delta = current_value - baseline_value

    if metric_name in LATENCY_METRICS or metric_name == "compression_ratio":
        if baseline_value == 0:
            return None
        delta_pct = (delta / baseline_value) * 100.0
        if metric_name in LATENCY_METRICS:
            is_regression = delta_pct > threshold_pct
            is_improvement = delta_pct < -threshold_pct
        else:
            is_regression = delta_pct < -threshold_pct
            is_improvement = delta_pct > threshold_pct
        if not (is_regression or is_improvement):
            return None
        return CellDiff(
            task=task,
            model=cell["model"],
            aggressiveness=cell["aggressiveness"],
            metric=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            delta=delta,
            delta_pct=delta_pct,
            is_regression=is_regression,
            is_improvement=is_improvement,
        )

    delta_pp = delta * 100.0
    is_regression = delta_pp < -threshold_pp
    is_improvement = delta_pp > threshold_pp
    if not (is_regression or is_improvement):
        return None
    return CellDiff(
        task=task,
        model=cell["model"],
        aggressiveness=cell["aggressiveness"],
        metric=metric_name,
        baseline_value=baseline_value,
        current_value=current_value,
        delta=delta,
        delta_pct=None,
        is_regression=is_regression,
        is_improvement=is_improvement,
    )
