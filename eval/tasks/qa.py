"""Extractive QA task.

This is the template task — get this right and the others follow the pattern.
The flow per (model, aggressiveness) cell:

    for sample in samples:
        compressed_context = ttc.compress(sample.context, aggr=...)
        prediction = llm.answer(compressed_context, sample.question)
        score = exact_match(prediction, gold), f1(prediction, gold)

We also run a baseline (no compression) once per sample, and a separate
``<ttc_safe>`` sub-experiment on a subset of samples — see ``run`` for the
contract.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from eval.client import TTCClient, TTCClientError
from eval.llm_backend import LLMBackend
from eval.tokenizers import count_tokens

from ._common import aggregate_with_ci, build_sweep_cell, select_best_cell
from .base import EvalTask, SamplePrediction, TaskResult, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parent.parent.parent / "datasets" / "qa_samples.json"

PROTECTED_SUBSET_FRACTION = 1.0

PRIMARY_METRIC = "f1"

@dataclass
class _QASample:
    sample_id: str
    context: str
    question: str
    answer: str
    domain: str

class QATask(EvalTask):
    """Extractive QA over noisy contexts (mix of finance + general).
    """

    def __init__(self, dataset_path: str | Path = DEFAULT_DATASET) -> None:
        self.dataset_path = Path(dataset_path)

    @property
    def name(self) -> str:
        return "qa"

    @property
    def description(self) -> str:
        return "Extractive QA on noisy contexts (finance + general). Scores: EM, F1."

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
            _QASample(
                sample_id=r["id"],
                context=r["context"],
                question=r["question"],
                answer=r["answer"],
                domain=r.get("domain", "general"),
            )
            for r in raw[:max_samples]
        ]

        baseline_preds = self._run_baseline(samples, llm)

        cells = []
        protected_results: dict[tuple[str, float], dict[str, float]] = {}

        rng = random.Random(42)
        n_protected = max(1, int(round(len(samples) * PROTECTED_SUBSET_FRACTION)))
        protected_ids = set(rng.sample([s.sample_id for s in samples], k=n_protected))

        for model in models:
            for aggr in aggressiveness_levels:
                cell_preds: list[SamplePrediction] = []
                protected_f1s: list[float] = []
                unprotected_f1s: list[float] = []

                for sample in samples:
                    pred = self._run_one_sample(
                        sample, client, llm, model=model, aggressiveness=aggr,
                        protected=False,
                    )
                    cell_preds.append(pred)
                    if sample.sample_id in protected_ids and not pred.failed:
                        unprotected_f1s.append(pred.metrics["f1"])

                    if sample.sample_id in protected_ids:
                        protected_pred = self._run_one_sample(
                            sample, client, llm, model=model, aggressiveness=aggr,
                            protected=True,
                        )
                        if not protected_pred.failed:
                            protected_f1s.append(protected_pred.metrics["f1"])

                cell = build_sweep_cell(
                    model=model,
                    aggressiveness=aggr,
                    metric_names=["em", "f1"],
                    predictions=cell_preds,
                    total_samples=len(samples),
                )
                cells.append(cell)

                if protected_f1s and unprotected_f1s:
                    protected_results[(model, aggr)] = {
                        "protected_f1": sum(protected_f1s) / len(protected_f1s),
                        "unprotected_f1": sum(unprotected_f1s) / len(unprotected_f1s),
                        "n_pairs": min(len(protected_f1s), len(unprotected_f1s)),
                    }

        baseline_metrics = {
            "em": aggregate_with_ci("em", baseline_preds),
            "f1": aggregate_with_ci("f1", baseline_preds),
        }
        baseline_score = baseline_metrics[PRIMARY_METRIC]["mean"]
        best = select_best_cell(cells, primary_metric=PRIMARY_METRIC, baseline_score=baseline_score)

        summary_stats: dict = {
            "primary_metric": PRIMARY_METRIC,
            "n_protected_samples": n_protected,
            "ttc_safe_results": [
                {
                    "model": k[0], "aggressiveness": k[1],
                    "protected_f1": v["protected_f1"],
                    "unprotected_f1": v["unprotected_f1"],
                    "n_pairs": v["n_pairs"],
                }
                for k, v in sorted(protected_results.items())
            ],
        }

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
            summary_stats=summary_stats,
        )

    def _run_baseline(
        self, samples: list[_QASample], llm: LLMBackend
    ) -> list[SamplePrediction]:
        """No compression, sends raw context straight to the LLM."""
        preds: list[SamplePrediction] = []
        for s in samples:
            from eval.metrics import exact_match, f1_token_overlap
            t0 = time.perf_counter()
            try:
                response = llm.answer(s.context, s.question)
                llm_lat = llm.last_call_latency_ms
            except Exception as exc:  # noqa: BLE001
                logger.warning("baseline LLM call failed", extra={"sample_id": s.sample_id, "error": str(exc)})
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
                metrics={
                    "em": exact_match(response, s.answer),
                    "f1": f1_token_overlap(response, s.answer),
                },
                compression_ratio=0.0,
                ttc_latency_ms=0.0,
                llm_latency_ms=llm_lat,
                e2e_latency_ms=e2e,
                ttc_cost_usd=0.0,
                llm_input_tokens=count_tokens(s.context, llm.model),
                llm_output_tokens=count_tokens(response, llm.model),
                extra={"prediction": response},
            ))
        return preds

    def _run_one_sample(
        self,
        sample: _QASample,
        client: TTCClient,
        llm: LLMBackend,
        *,
        model: str,
        aggressiveness: float,
        protected: bool,
    ) -> SamplePrediction:
        from eval.metrics import exact_match, f1_token_overlap

        t0 = time.perf_counter()
        if protected:
            input_text = f"<ttc_safe>Question: {sample.question}</ttc_safe>\n\n{sample.context}"
        else:
            input_text = sample.context

        try:
            comp = client.compress(input_text, aggressiveness=aggressiveness, model=model)
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
            metrics={
                "em": exact_match(response, sample.answer),
                "f1": f1_token_overlap(response, sample.answer),
            },
            compression_ratio=comp.compression_ratio,
            ttc_latency_ms=comp.latency_ms,
            llm_latency_ms=llm_lat,
            e2e_latency_ms=e2e,
            ttc_cost_usd=comp.ttc_cost_usd,
            llm_input_tokens=count_tokens(comp.output, llm.model),
            llm_output_tokens=count_tokens(response, llm.model),
            extra={"prediction": response, "protected": protected},
        )

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
