"""Regression detection: synthetic baseline + planted regression must be flagged."""

from __future__ import annotations

from eval.regression import compare_runs


def _make_cell(model: str, aggr: float, f1_mean: float, comp_ratio: float, p95_e2e: float) -> dict:
    return {
        "model": model,
        "aggressiveness": aggr,
        "metrics": {"f1": {"mean": f1_mean, "ci_low": f1_mean - 0.05, "ci_high": f1_mean + 0.05}},
        "avg_compression_ratio": comp_ratio,
        "p95_e2e_latency_ms": p95_e2e,
        "p95_llm_latency_ms": p95_e2e * 0.8,
        "p95_ttc_latency_ms": p95_e2e * 0.2,
    }

def _make_report(cells: list[dict]) -> dict:
    return {
        "tasks": [
            {
                "task_name": "qa",
                "sweep_results": cells,
            }
        ]
    }

def test_planted_accuracy_regression_is_detected() -> None:
    baseline = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.80, comp_ratio=0.4, p95_e2e=300)])
    current = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.75, comp_ratio=0.4, p95_e2e=300)])

    report = compare_runs(current, baseline, threshold_pp=1.0)
    assert report.summary["regressions_found"] >= 1
    f1_diffs = [d for d in report.regressions if d.metric == "f1"]
    assert len(f1_diffs) == 1
    assert f1_diffs[0].is_regression
    assert f1_diffs[0].delta < 0

def test_compression_ratio_regression_is_flagged() -> None:
    baseline = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.80, comp_ratio=0.40, p95_e2e=300)])
    current = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.80, comp_ratio=0.30, p95_e2e=300)])

    report = compare_runs(current, baseline, threshold_pct=5.0)
    comp_regressions = [d for d in report.regressions if d.metric == "compression_ratio"]
    assert len(comp_regressions) == 1

def test_latency_regression_is_flagged() -> None:
    baseline = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.80, comp_ratio=0.40, p95_e2e=300)])
    current = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.80, comp_ratio=0.40, p95_e2e=600)])

    report = compare_runs(current, baseline, threshold_pct=10.0)
    lat_regressions = [d for d in report.regressions if d.metric == "p95_e2e_latency_ms"]
    assert len(lat_regressions) == 1

def test_no_change_means_no_regressions() -> None:
    cell = _make_cell("bear-1.2", 0.5, f1_mean=0.80, comp_ratio=0.4, p95_e2e=300)
    report = compare_runs(_make_report([cell]), _make_report([cell]), threshold_pp=1.0)
    assert report.summary["regressions_found"] == 0
    assert report.summary["improvements_found"] == 0

def test_improvement_is_recorded_separately() -> None:
    baseline = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.70, comp_ratio=0.40, p95_e2e=300)])
    current = _make_report([_make_cell("bear-1.2", 0.5, f1_mean=0.78, comp_ratio=0.40, p95_e2e=300)])

    report = compare_runs(current, baseline, threshold_pp=1.0)
    f1_improvements = [d for d in report.improvements if d.metric == "f1"]
    assert len(f1_improvements) == 1
    assert f1_improvements[0].is_improvement
