"""RAG retrieval-quality task.

For each sample we have a query, a list of 5 candidate documents, and the
index of the relevant one. We compress each document independently, then
rank by semantic similarity to the query. Metrics:

* ``retrieval_accuracy``: does the relevant document still rank #1?
* ``mrr``: mean reciprocal rank of the relevant doc in the ranking.

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from eval.client import TTCClient, TTCClientError
from eval.llm_backend import LLMBackend
from eval.metrics import semantic_similarity
from eval.tokenizers import count_tokens

from ._common import aggregate_with_ci, build_sweep_cell, select_best_cell
from .base import EvalTask, SamplePrediction, TaskResult, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent.parent / "datasets" / "rag_samples.json"
)
PRIMARY_METRIC = "retrieval_accuracy"

@dataclass
class _RAGSample:
    sample_id: str
    documents: list[str]
    query: str
    relevant_doc_index: int

class RAGTask(EvalTask):
    """Retrieval ranking over compressed candidate documents."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET,
        use_embeddings: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.use_embeddings = use_embeddings

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return (
            "RAG retrieval ranking — does the relevant doc still rank #1 after compression? "
            "Scores: retrieval_accuracy, MRR."
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
            _RAGSample(
                sample_id=r["id"],
                documents=r["documents"],
                query=r["query"],
                relevant_doc_index=int(r["relevant_doc_index"]),
            )
            for r in raw[:max_samples]
        ]

        baseline_preds = self._run_baseline(samples, llm)
        cells = []
        for model in models:
            for aggr in aggressiveness_levels:
                cell_preds = [
                    self._run_one_sample(s, client, llm, model=model, aggressiveness=aggr)
                    for s in samples
                ]
                cell = build_sweep_cell(
                    model=model,
                    aggressiveness=aggr,
                    metric_names=["retrieval_accuracy", "mrr"],
                    predictions=cell_preds,
                    total_samples=len(samples),
                )
                cells.append(cell)

        baseline_metrics = {
            "retrieval_accuracy": aggregate_with_ci("retrieval_accuracy", baseline_preds),
            "mrr": aggregate_with_ci("mrr", baseline_preds),
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
            summary_stats={"primary_metric": PRIMARY_METRIC, "use_embeddings": self.use_embeddings},
        )

    def _run_baseline(
        self, samples: list[_RAGSample], llm: LLMBackend
    ) -> list[SamplePrediction]:
        preds: list[SamplePrediction] = []
        for s in samples:
            t0 = time.perf_counter()
            ranking = self._rank(s.documents, s.query)
            elapsed = (time.perf_counter() - t0) * 1000.0
            acc, mrr = self._score_ranking(ranking, s.relevant_doc_index)
            preds.append(SamplePrediction(
                sample_id=s.sample_id,
                metrics={"retrieval_accuracy": acc, "mrr": mrr},
                compression_ratio=0.0,
                ttc_latency_ms=0.0,
                llm_latency_ms=elapsed,
                e2e_latency_ms=elapsed,
                ttc_cost_usd=0.0,
                llm_input_tokens=sum(count_tokens(d, llm.model) for d in s.documents),
                llm_output_tokens=0,
                extra={"ranking": ranking},
            ))
        return preds

    def _run_one_sample(
        self,
        sample: _RAGSample,
        client: TTCClient,
        llm: LLMBackend,
        *,
        model: str,
        aggressiveness: float,
    ) -> SamplePrediction:
        t0 = time.perf_counter()
        compressed_docs: list[str] = []
        ttc_latency_total = 0.0
        ttc_cost_total = 0.0
        compression_ratios: list[float] = []
        try:
            for doc in sample.documents:
                comp = client.compress(doc, aggressiveness=aggressiveness, model=model)
                compressed_docs.append(comp.output)
                ttc_latency_total += comp.latency_ms
                ttc_cost_total += comp.ttc_cost_usd
                compression_ratios.append(comp.compression_ratio)
        except TTCClientError as exc:
            return SamplePrediction(
                sample_id=sample.sample_id, metrics={}, compression_ratio=0.0,
                ttc_latency_ms=ttc_latency_total, llm_latency_ms=0.0, e2e_latency_ms=0.0,
                ttc_cost_usd=ttc_cost_total, llm_input_tokens=0, llm_output_tokens=0,
                failed=True, failure_reason=f"TTC failed: {exc}",
            )

        rank_start = time.perf_counter()
        ranking = self._rank(compressed_docs, sample.query)
        rank_lat = (time.perf_counter() - rank_start) * 1000.0
        e2e = (time.perf_counter() - t0) * 1000.0
        acc, mrr = self._score_ranking(ranking, sample.relevant_doc_index)
        avg_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0

        return SamplePrediction(
            sample_id=sample.sample_id,
            metrics={"retrieval_accuracy": acc, "mrr": mrr},
            compression_ratio=avg_ratio,
            ttc_latency_ms=ttc_latency_total,
            llm_latency_ms=rank_lat,
            e2e_latency_ms=e2e,
            ttc_cost_usd=ttc_cost_total,
            llm_input_tokens=sum(count_tokens(d, llm.model) for d in compressed_docs),
            llm_output_tokens=0,
            extra={"ranking": ranking},
        )

    def _rank(self, documents: list[str], query: str) -> list[int]:
        scored = [
            (semantic_similarity(d, query, use_embeddings=self.use_embeddings), idx)
            for idx, d in enumerate(documents)
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [idx for _, idx in scored]

    @staticmethod
    def _score_ranking(ranking: list[int], gold_idx: int) -> tuple[float, float]:
        accuracy = 1.0 if ranking and ranking[0] == gold_idx else 0.0
        try:
            position = ranking.index(gold_idx)
            mrr = 1.0 / (position + 1)
        except ValueError:
            mrr = 0.0
        return accuracy, mrr

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
