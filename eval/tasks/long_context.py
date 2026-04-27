"""Long-context needle-in-a-haystack — the headline chart of the report.

Compress a long haystack, then ask a question whose answer is a single
sentence buried somewhere in it. The 2D heatmap of needle-position ×
aggressiveness is the strongest single argument for the product: it shows,
visually, how aggressively you can compress before a hidden fact starts to
disappear.

The needle position takes one of {start, middle, end}. We track per-position
retrieval accuracy and surface it in summary_stats so the report's heatmap
has what it needs.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from eval.client import TTCClient, TTCClientError
from eval.llm_backend import LLMBackend
from eval.metrics import f1_token_overlap
from eval.tokenizers import count_tokens

from ._common import aggregate_with_ci, build_sweep_cell, select_best_cell
from .base import EvalTask, SamplePrediction, TaskResult, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent.parent / "datasets" / "needle_samples.json"
)
PRIMARY_METRIC = "retrieval_accuracy"

@dataclass
class _NeedleSample:
    sample_id: str
    haystack: str
    needle: str
    needle_position: str
    question: str
    answer: str

class LongContextNeedleTask(EvalTask):
    """Needle-in-a-haystack retrieval after compression."""

    def __init__(self, dataset_path: str | Path = DEFAULT_DATASET) -> None:
        self.dataset_path = Path(dataset_path)

    @property
    def name(self) -> str:
        return "long_context"

    @property
    def description(self) -> str:
        return (
            "Long-context needle-in-haystack. Heatmap of retrieval accuracy "
            "across needle position × aggressiveness — the headline chart."
        )

    def run(
        self,
        client: TTCClient,
        llm: LLMBackend,
        models: list[str],
        aggressiveness_levels: list[float],
        max_samples: int,
    ) -> TaskResult:
        raw = load_dataset(self.dataset_path)
        samples = [
            _NeedleSample(
                sample_id=r["id"],
                haystack=r["haystack"],
                needle=r["needle"],
                needle_position=r["needle_position"],
                question=r["question"],
                answer=r["answer"],
            )
            for r in raw[:max_samples]
        ]

        baseline_preds = self._run_baseline(samples, llm)
        cells = []
        heatmap_stats: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for model in models:
            for aggr in aggressiveness_levels:
                cell_preds = []
                for s in samples:
                    pred = self._run_one_sample(s, client, llm, model=model, aggressiveness=aggr)
                    cell_preds.append(pred)
                    if not pred.failed:
                        heatmap_stats[(model, aggr)][s.needle_position].append(
                            pred.metrics["retrieval_accuracy"]
                        )
                cell = build_sweep_cell(
                    model=model,
                    aggressiveness=aggr,
                    metric_names=["retrieval_accuracy", "f1"],
                    predictions=cell_preds,
                    total_samples=len(samples),
                )
                cells.append(cell)

        baseline_metrics = {
            "retrieval_accuracy": aggregate_with_ci("retrieval_accuracy", baseline_preds),
            "f1": aggregate_with_ci("f1", baseline_preds),
        }
        baseline_score = baseline_metrics[PRIMARY_METRIC]["mean"]
        best = select_best_cell(cells, primary_metric=PRIMARY_METRIC, baseline_score=baseline_score)

        heatmap_rows = []
        for (model, aggr), pos_map in heatmap_stats.items():
            for position, accs in pos_map.items():
                if not accs:
                    continue
                heatmap_rows.append({
                    "model": model,
                    "aggressiveness": aggr,
                    "needle_position": position,
                    "retrieval_accuracy": sum(accs) / len(accs),
                    "n": len(accs),
                })

        return TaskResult(
            task_name=self.name,
            description=self.description,
            samples_tested=len(samples),
            baseline_metrics=baseline_metrics,
            baseline_avg_llm_input_tokens=_mean(p.llm_input_tokens for p in baseline_preds),
            baseline_avg_llm_output_tokens=_mean(p.llm_output_tokens for p in baseline_preds),
            baseline_avg_llm_latency_ms=_mean(p.llm_latency_ms for p in baseline_preds),
            baseline_p95_llm_latency_ms=_p95([p.llm_latency_ms for p in baseline_preds]),
            baseline_failure_count=sum(1 for p in baseline_preds if p.failed),
            sweep_results=cells,
            best_cell=best,
            summary_stats={
                "primary_metric": PRIMARY_METRIC,
                "heatmap": heatmap_rows,
            },
        )

    def _run_baseline(
        self, samples: list[_NeedleSample], llm: LLMBackend
    ) -> list[SamplePrediction]:
        preds = []
        for s in samples:
            t0 = time.perf_counter()
            try:
                response = llm.answer(s.haystack, s.question)
                llm_lat = llm.last_call_latency_ms
            except Exception as exc:  # noqa: BLE001
                preds.append(SamplePrediction(
                    sample_id=s.sample_id, metrics={}, compression_ratio=0.0,
                    ttc_latency_ms=0.0, llm_latency_ms=0.0, e2e_latency_ms=0.0,
                    ttc_cost_usd=0.0, llm_input_tokens=0, llm_output_tokens=0,
                    failed=True, failure_reason=f"baseline LLM failed: {exc}",
                ))
                continue
            e2e = (time.perf_counter() - t0) * 1000.0
            preds.append(SamplePrediction(
                sample_id=s.sample_id,
                metrics=self._score(response, s.answer),
                compression_ratio=0.0,
                ttc_latency_ms=0.0,
                llm_latency_ms=llm_lat,
                e2e_latency_ms=e2e,
                ttc_cost_usd=0.0,
                llm_input_tokens=count_tokens(s.haystack, llm.model),
                llm_output_tokens=count_tokens(response, llm.model),
                extra={"prediction": response, "needle_position": s.needle_position},
            ))
        return preds

    def _run_one_sample(
        self,
        sample: _NeedleSample,
        client: TTCClient,
        llm: LLMBackend,
        *,
        model: str,
        aggressiveness: float,
    ) -> SamplePrediction:
        t0 = time.perf_counter()
        try:
            comp = client.compress(sample.haystack, aggressiveness=aggressiveness, model=model)
        except TTCClientError as exc:
            return SamplePrediction(
                sample_id=sample.sample_id, metrics={}, compression_ratio=0.0,
                ttc_latency_ms=0.0, llm_latency_ms=0.0, e2e_latency_ms=0.0,
                ttc_cost_usd=0.0, llm_input_tokens=0, llm_output_tokens=0,
                failed=True, failure_reason=f"TTC failed: {exc}",
            )

        try:
            response = llm.answer(comp.output, sample.question)
            llm_lat = llm.last_call_latency_ms
        except Exception as exc:  # noqa: BLE001
            return SamplePrediction(
                sample_id=sample.sample_id, metrics={}, compression_ratio=comp.compression_ratio,
                ttc_latency_ms=comp.latency_ms, llm_latency_ms=0.0, e2e_latency_ms=0.0,
                ttc_cost_usd=comp.ttc_cost_usd, llm_input_tokens=0, llm_output_tokens=0,
                failed=True, failure_reason=f"LLM failed: {exc}",
            )

        e2e = (time.perf_counter() - t0) * 1000.0
        return SamplePrediction(
            sample_id=sample.sample_id,
            metrics=self._score(response, sample.answer),
            compression_ratio=comp.compression_ratio,
            ttc_latency_ms=comp.latency_ms,
            llm_latency_ms=llm_lat,
            e2e_latency_ms=e2e,
            ttc_cost_usd=comp.ttc_cost_usd,
            llm_input_tokens=count_tokens(comp.output, llm.model),
            llm_output_tokens=count_tokens(response, llm.model),
            extra={"prediction": response, "needle_position": sample.needle_position},
        )

    @staticmethod
    def _score(prediction: str, gold: str) -> dict[str, float]:
        return {
            "retrieval_accuracy": 1.0 if gold.lower() in prediction.lower() else 0.0,
            "f1": f1_token_overlap(prediction, gold),
        }

def _mean(values) -> float:  # type: ignore[no-untyped-def]
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0

def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = 0.95 * (len(sorted_vals) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)
