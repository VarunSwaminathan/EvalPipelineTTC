"""Abstractive summarization with a faithfulness check.

Faithfulness is measured against the **original** (uncompressed) document
even when the LLM saw the compressed version — the question we're asking is
"does the summary still represent source truth after compression?"

The report cross-plots faithfulness vs ROUGE-L per sample. The interesting
quadrant is high-ROUGE-low-faithfulness: compression-induced confabulation,
where the summary reads close to the reference but invents details. The
report should call these out by sample id.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from eval.client import TTCClient, TTCClientError
from eval.llm_backend import LLMBackend
from eval.metrics import faithfulness_score, rouge_l
from eval.tokenizers import count_tokens

from ._common import aggregate_with_ci, build_sweep_cell, select_best_cell
from .base import EvalTask, SamplePrediction, TaskResult, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent.parent / "datasets" / "summarization_samples.json"
)
PRIMARY_METRIC = "rouge_l"

@dataclass
class _SumSample:
    sample_id: str
    document: str
    reference_summary: str
    domain: str

class SummarizationTask(EvalTask):
    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET,
        use_embeddings: bool = False,
        max_words: int = 100,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.use_embeddings = use_embeddings
        self.max_words = max_words

    @property
    def name(self) -> str:
        return "summarization"

    @property
    def description(self) -> str:
        return (
            "Abstractive summarization. Scores: ROUGE-L vs reference and "
            "faithfulness vs the *original* document."
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
            _SumSample(
                sample_id=r["id"],
                document=r["document"],
                reference_summary=r["reference_summary"],
                domain=r.get("domain", "general"),
            )
            for r in raw[:max_samples]
        ]

        baseline_preds = self._run_baseline(samples, llm)
        cells = []
        confabulations: list[dict] = []
        for model in models:
            for aggr in aggressiveness_levels:
                cell_preds = [
                    self._run_one_sample(s, client, llm, model=model, aggressiveness=aggr)
                    for s in samples
                ]
                cell = build_sweep_cell(
                    model=model,
                    aggressiveness=aggr,
                    metric_names=["rouge_l", "faithfulness"],
                    predictions=cell_preds,
                    total_samples=len(samples),
                )
                cells.append(cell)

                for p in cell_preds:
                    if p.failed:
                        continue
                    rg = p.metrics.get("rouge_l", 0.0)
                    fa = p.metrics.get("faithfulness", 1.0)
                    if rg >= 0.5 and fa < 0.5:
                        confabulations.append({
                            "sample_id": p.sample_id,
                            "model": model,
                            "aggressiveness": aggr,
                            "rouge_l": rg,
                            "faithfulness": fa,
                        })

        baseline_metrics = {
            "rouge_l": aggregate_with_ci("rouge_l", baseline_preds),
            "faithfulness": aggregate_with_ci("faithfulness", baseline_preds),
        }
        baseline_score = baseline_metrics[PRIMARY_METRIC]["mean"]
        best = select_best_cell(cells, primary_metric=PRIMARY_METRIC, baseline_score=baseline_score)

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
                "confabulations": confabulations,
                "use_embeddings": self.use_embeddings,
            },
        )

    def _run_baseline(self, samples: list[_SumSample], llm: LLMBackend) -> list[SamplePrediction]:
        preds: list[SamplePrediction] = []
        for s in samples:
            t0 = time.perf_counter()
            try:
                summary = llm.summarize(s.document, max_words=self.max_words)
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
                metrics={
                    "rouge_l": rouge_l(summary, s.reference_summary),
                    "faithfulness": faithfulness_score(
                        summary, s.document, use_embeddings=self.use_embeddings
                    ),
                },
                compression_ratio=0.0,
                ttc_latency_ms=0.0,
                llm_latency_ms=llm_lat,
                e2e_latency_ms=e2e,
                ttc_cost_usd=0.0,
                llm_input_tokens=count_tokens(s.document, llm.model),
                llm_output_tokens=count_tokens(summary, llm.model),
                extra={"summary": summary},
            ))
        return preds

    def _run_one_sample(
        self,
        sample: _SumSample,
        client: TTCClient,
        llm: LLMBackend,
        *,
        model: str,
        aggressiveness: float,
    ) -> SamplePrediction:
        t0 = time.perf_counter()
        try:
            comp = client.compress(sample.document, aggressiveness=aggressiveness, model=model)
        except TTCClientError as exc:
            return SamplePrediction(
                sample_id=sample.sample_id, metrics={}, compression_ratio=0.0,
                ttc_latency_ms=0.0, llm_latency_ms=0.0, e2e_latency_ms=0.0,
                ttc_cost_usd=0.0, llm_input_tokens=0, llm_output_tokens=0,
                failed=True, failure_reason=f"TTC failed: {exc}",
            )

        try:
            summary = llm.summarize(comp.output, max_words=self.max_words)
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
                "rouge_l": rouge_l(summary, sample.reference_summary),
                "faithfulness": faithfulness_score(
                    summary, sample.document, use_embeddings=self.use_embeddings
                ),
            },
            compression_ratio=comp.compression_ratio,
            ttc_latency_ms=comp.latency_ms,
            llm_latency_ms=llm_lat,
            e2e_latency_ms=e2e,
            ttc_cost_usd=comp.ttc_cost_usd,
            llm_input_tokens=count_tokens(comp.output, llm.model),
            llm_output_tokens=count_tokens(summary, llm.model),
            extra={"summary": summary},
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
