"""Abstract task base + result containers shared across all tasks.

The whole eval is structured as a sweep across (task × model × aggressiveness).
Each ``EvalTask`` knows how to execute one task across all (model, aggr)
cells. The runner orchestrates tasks. This keeps task-specific logic (prompt
shapes, scoring choices, sub-experiments like the ``<ttc_safe>`` test) inside
the task file, and keeps the runner generic.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.client import TTCClient
from eval.llm_backend import LLMBackend


@dataclass
class SamplePrediction:
    """Per-sample raw record. The full set goes to JSON for re-rendering reports."""

    sample_id: str
    metrics: dict[str, float]
    compression_ratio: float
    ttc_latency_ms: float
    llm_latency_ms: float
    e2e_latency_ms: float
    ttc_cost_usd: float
    llm_input_tokens: int
    llm_output_tokens: int
    failed: bool = False
    failure_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class SweepCell:
    """One (model, aggressiveness) cell's aggregated results.

    ``baseline_metrics`` is intentionally not stored on the cell, it lives at
    the TaskResult level since it doesn't depend on (model, aggressiveness).
    """

    model: str
    aggressiveness: float

    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    avg_compression_ratio: float = 0.0
    avg_ttc_cost_usd: float = 0.0
    avg_ttc_latency_ms: float = 0.0
    p50_ttc_latency_ms: float = 0.0
    p95_ttc_latency_ms: float = 0.0
    p99_ttc_latency_ms: float = 0.0

    avg_llm_input_tokens: float = 0.0
    avg_llm_output_tokens: float = 0.0
    avg_llm_latency_ms: float = 0.0
    p50_llm_latency_ms: float = 0.0
    p95_llm_latency_ms: float = 0.0
    p99_llm_latency_ms: float = 0.0

    avg_e2e_latency_ms: float = 0.0
    p50_e2e_latency_ms: float = 0.0
    p95_e2e_latency_ms: float = 0.0
    p99_e2e_latency_ms: float = 0.0

    failure_count: int = 0
    failure_rate: float = 0.0

    sample_predictions: list[SamplePrediction] = field(default_factory=list)

@dataclass
class TaskResult:
    """Everything one task produced. Persisted to JSON for the report."""

    task_name: str
    description: str
    samples_tested: int
    baseline_metrics: dict[str, dict[str, float]]
    baseline_avg_llm_input_tokens: float
    baseline_avg_llm_output_tokens: float
    baseline_avg_llm_latency_ms: float
    baseline_p95_llm_latency_ms: float
    baseline_failure_count: int
    sweep_results: list[SweepCell]
    best_cell: SweepCell | None
    summary_stats: dict[str, Any] = field(default_factory=dict)

class EvalTask(ABC):
    """Tasks load their own dataset and own their scoring logic.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def run(
        self,
        client: TTCClient,
        llm: LLMBackend,
        models: list[str],
        aggressiveness_levels: list[float],
        max_samples: int,
    ) -> TaskResult: ...

def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def task_result_to_dict(result: TaskResult) -> dict[str, Any]:
    """``asdict`` plus a couple of cleanups for JSON serialization."""
    return asdict(result)
