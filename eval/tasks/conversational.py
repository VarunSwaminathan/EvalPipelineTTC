"""Multi-turn conversational QA (CoQA-style).

The passage is compressed once. The dialogue then runs against the compressed
passage, with each turn appending all prior Q/A pairs (standard CoQA setup).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from eval.client import TTCClient, TTCClientError
from eval.llm_backend import LLMBackend
from eval.metrics import exact_match, f1_token_overlap
from eval.tokenizers import count_tokens

from ._common import aggregate_with_ci, build_sweep_cell, select_best_cell
from .base import EvalTask, SamplePrediction, TaskResult, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent.parent / "datasets" / "conversational_samples.json"
)
PRIMARY_METRIC = "f1"

@dataclass
class _ConvSample:
    sample_id: str
    passage: str
    turns: list[dict[str, str]]

class ConversationalQATask(EvalTask):
    """Multi-turn QA with prior turns concatenated into each new prompt."""

    def __init__(self, dataset_path: str | Path = DEFAULT_DATASET) -> None:
        self.dataset_path = Path(dataset_path)

    @property
    def name(self) -> str:
        return "conversational"

    @property
    def description(self) -> str:
        return (
            "Multi-turn conversational QA. Compresses the passage once, then "
            "answers a sequence of dependent questions. Reports per-turn-position "
            "accuracy to surface degradation deeper in the dialogue."
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
            _ConvSample(
                sample_id=r["id"],
                passage=r["passage"],
                turns=r["turns"],
            )
            for r in raw[:max_samples]
        ]

        baseline_preds = self._run_baseline(samples, llm)
        cells = []
        turn_position_stats: dict[tuple[str, float], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for model in models:
            for aggr in aggressiveness_levels:
                cell_preds: list[SamplePrediction] = []
                for s in samples:
                    pred = self._run_one_sample(s, client, llm, model=model, aggressiveness=aggr)
                    cell_preds.append(pred)
                    if not pred.failed:
                        for turn_idx, turn_f1 in enumerate(pred.extra.get("turn_f1s", [])):
                            turn_position_stats[(model, aggr)][turn_idx].append(turn_f1)
                cell = build_sweep_cell(
                    model=model,
                    aggressiveness=aggr,
                    metric_names=["em", "f1", "conversation_success"],
                    predictions=cell_preds,
                    total_samples=len(samples),
                )
                cells.append(cell)

        baseline_metrics = {
            "em": aggregate_with_ci("em", baseline_preds),
            "f1": aggregate_with_ci("f1", baseline_preds),
            "conversation_success": aggregate_with_ci("conversation_success", baseline_preds),
        }
        baseline_score = baseline_metrics[PRIMARY_METRIC]["mean"]
        best = select_best_cell(cells, primary_metric=PRIMARY_METRIC, baseline_score=baseline_score)

        per_turn_summary = []
        for (model, aggr), idx_map in turn_position_stats.items():
            for turn_idx, f1s in sorted(idx_map.items()):
                if not f1s:
                    continue
                per_turn_summary.append({
                    "model": model,
                    "aggressiveness": aggr,
                    "turn_index": turn_idx,
                    "mean_f1": sum(f1s) / len(f1s),
                    "n": len(f1s),
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
                "per_turn_accuracy": per_turn_summary,
            },
        )

    def _run_baseline(
        self, samples: list[_ConvSample], llm: LLMBackend
    ) -> list[SamplePrediction]:
        return [self._dialogue(s, passage=s.passage, llm=llm, baseline=True) for s in samples]

    def _run_one_sample(
        self,
        sample: _ConvSample,
        client: TTCClient,
        llm: LLMBackend,
        *,
        model: str,
        aggressiveness: float,
    ) -> SamplePrediction:
        try:
            comp = client.compress(sample.passage, aggressiveness=aggressiveness, model=model)
        except TTCClientError as exc:
            return SamplePrediction(
                sample_id=sample.sample_id, metrics={}, compression_ratio=0.0,
                ttc_latency_ms=0.0, llm_latency_ms=0.0, e2e_latency_ms=0.0,
                ttc_cost_usd=0.0, llm_input_tokens=0, llm_output_tokens=0,
                failed=True, failure_reason=f"TTC failed: {exc}",
            )
        return self._dialogue(
            sample,
            passage=comp.output,
            llm=llm,
            baseline=False,
            ttc_latency_ms=comp.latency_ms,
            ttc_cost_usd=comp.ttc_cost_usd,
            compression_ratio=comp.compression_ratio,
        )

    def _dialogue(
        self,
        sample: _ConvSample,
        *,
        passage: str,
        llm: LLMBackend,
        baseline: bool,
        ttc_latency_ms: float = 0.0,
        ttc_cost_usd: float = 0.0,
        compression_ratio: float = 0.0,
    ) -> SamplePrediction:
        """Run all turns of a dialogue, return one aggregate SamplePrediction."""
        t0 = time.perf_counter()
        ems: list[float] = []
        f1s: list[float] = []
        history_lines: list[str] = []
        total_llm_lat = 0.0
        total_in_tokens = 0
        total_out_tokens = 0
        responses: list[str] = []
        try:
            for turn in sample.turns:
                history_block = "\n".join(history_lines) if history_lines else ""
                full_context = (
                    f"Passage:\n{passage}\n\n"
                    + (f"Previous turns:\n{history_block}\n\n" if history_block else "")
                )
                response = llm.answer(full_context, turn["question"])
                total_llm_lat += llm.last_call_latency_ms
                total_in_tokens += count_tokens(full_context, llm.model)
                total_out_tokens += count_tokens(response, llm.model)

                em = exact_match(response, turn["answer"])
                f1 = f1_token_overlap(response, turn["answer"])
                ems.append(em)
                f1s.append(f1)
                responses.append(response)

                history_lines.append(f"Q: {turn['question']}\nA: {turn['answer']}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "conversation failed mid-dialogue",
                extra={"sample_id": sample.sample_id, "error": str(exc), "baseline": baseline},
            )
            return SamplePrediction(
                sample_id=sample.sample_id, metrics={}, compression_ratio=compression_ratio,
                ttc_latency_ms=ttc_latency_ms, llm_latency_ms=total_llm_lat, e2e_latency_ms=0.0,
                ttc_cost_usd=ttc_cost_usd, llm_input_tokens=total_in_tokens,
                llm_output_tokens=total_out_tokens,
                failed=True, failure_reason=f"LLM dialogue failed: {exc}",
            )

        e2e = (time.perf_counter() - t0) * 1000.0
        success = 1.0 if all(em == 1.0 for em in ems) else 0.0
        return SamplePrediction(
            sample_id=sample.sample_id,
            metrics={
                "em": sum(ems) / len(ems) if ems else 0.0,
                "f1": sum(f1s) / len(f1s) if f1s else 0.0,
                "conversation_success": success,
            },
            compression_ratio=compression_ratio,
            ttc_latency_ms=ttc_latency_ms,
            llm_latency_ms=total_llm_lat,
            e2e_latency_ms=e2e,
            ttc_cost_usd=ttc_cost_usd,
            llm_input_tokens=total_in_tokens,
            llm_output_tokens=total_out_tokens,
            extra={"turn_f1s": f1s, "responses": responses},
        )

def _mean(values) -> float:
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
