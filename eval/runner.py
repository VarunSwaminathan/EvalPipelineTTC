"""EvalRunner — orchestrates tasks across the (model × aggressiveness) sweep.

Responsibilities:

* Run an idempotence pre-check against the live TTC API and log a warning
  (not a hard failure) if outputs differ. The TTC docs describe the models
  as deterministic; we want to know quickly if that ever stops being true.
* Execute each task in sequence, with per-sample work parallelized by the
  task itself if it elects to.
* Print live progress with ``rich``.
* Persist the full ``EvalReport`` JSON so the HTML report can be regenerated
  from saved results without re-running the eval.
* Run regression diff against a baseline if one was provided.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from eval.client import TTCClient
from eval.llm_backend import LLMBackend
from eval.regression import RegressionReport, compare_runs, load_report_json
from eval.tasks.base import EvalTask, TaskResult, task_result_to_dict

logger = logging.getLogger(__name__)

IDEMPOTENCE_PROBE_TEXT = (
    "Acme Industries reported total net revenue of $12.4 billion for fiscal "
    "year 2023, an increase of 8.2 percent year-over-year, driven primarily "
    "by strong demand in the industrial automation segment, which grew 14 "
    "percent. The company noted that gross margin contracted modestly to 38.1 "
    "percent from 39.4 percent in the prior year, reflecting elevated input "
    "costs and ongoing supply-chain headwinds that management characterized "
    "as transitory. Operating income came in at $2.1 billion."
)

@dataclass
class EvalReport:
    """Top-level run record. Persisted as JSON for re-rendering."""

    run_id: str
    started_at: str
    finished_at: str
    duration_s: float
    backend_name: str
    backend_model: str
    models: list[str]
    aggressiveness_levels: list[float]
    max_samples: int
    seed: int
    use_embeddings: bool
    idempotence_check: dict[str, Any]
    tasks: list[dict[str, Any]] = field(default_factory=list)
    regression: dict[str, Any] | None = None

class EvalRunner:
    def __init__(
        self,
        client: TTCClient,
        llm: LLMBackend,
        tasks: list[EvalTask],
        models: list[str],
        aggressiveness_levels: list[float],
        max_workers: int = 4,
        seed: int = 42,
        use_embeddings: bool = False,
        output_dir: str | Path = "results",
        baseline_path: str | Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.client = client
        self.llm = llm
        self.tasks = tasks
        self.models = models
        self.aggressiveness_levels = aggressiveness_levels
        self.max_workers = max_workers
        self.seed = seed
        self.use_embeddings = use_embeddings
        self.output_dir = Path(output_dir)
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.console = console or Console()

    def run(self, max_samples: int = 50) -> tuple[EvalReport, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()

        idempotence = self._idempotence_check()

        task_results: list[TaskResult] = []
        for task in self.tasks:
            self.console.rule(f"[bold cyan]{task.name}[/]")
            self.console.print(f"[dim]{task.description}[/dim]")
            tic = time.perf_counter()
            result = task.run(
                client=self.client,
                llm=self.llm,
                models=self.models,
                aggressiveness_levels=self.aggressiveness_levels,
                max_samples=max_samples,
            )
            elapsed = time.perf_counter() - tic
            self.console.print(
                f"[green][OK] {task.name}[/] [dim]({result.samples_tested} samples · "
                f"{len(result.sweep_results)} cells · {elapsed:.1f}s)[/]"
            )
            self._print_task_summary(result)
            task_results.append(result)

        finished = datetime.now(timezone.utc).isoformat()
        duration = time.perf_counter() - t0
        report = EvalReport(
            run_id=run_id,
            started_at=started,
            finished_at=finished,
            duration_s=duration,
            backend_name=self.llm.name,
            backend_model=self.llm.model,
            models=self.models,
            aggressiveness_levels=self.aggressiveness_levels,
            max_samples=max_samples,
            seed=self.seed,
            use_embeddings=self.use_embeddings,
            idempotence_check=idempotence,
            tasks=[task_result_to_dict(r) for r in task_results],
        )

        if self.baseline_path:
            try:
                baseline = load_report_json(self.baseline_path)
                regression = compare_runs(asdict(report), baseline)
                report.regression = regression.to_dict()
                self._print_regression_summary(regression)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "regression comparison failed",
                    extra={"baseline": str(self.baseline_path), "error": str(exc)},
                )

        out_path = self.output_dir / f"results_{run_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        self.console.print(f"\n[bold]Saved:[/] {out_path}")
        return report, out_path

    def _idempotence_check(self) -> dict[str, Any]:
        """TTC describes its models as deterministic. Verify that, log on drift.

        We do NOT hard-fail on a mismatch — flaky cell results would block
        eval runs unnecessarily. Surfacing the discrepancy in the report
        and the logs is the right behavior for an internal tool.
        """
        self.console.rule("[dim]idempotence pre-check[/]")
        try:
            a = self.client.compress(IDEMPOTENCE_PROBE_TEXT, aggressiveness=0.5)
            b = self.client.compress(IDEMPOTENCE_PROBE_TEXT, aggressiveness=0.5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("idempotence pre-check failed to execute", extra={"error": str(exc)})
            self.console.print(f"[yellow]idempotence check failed to run:[/] {exc}")
            return {"executed": False, "matched": None, "error": str(exc)}

        matched = a.output == b.output and a.output_tokens == b.output_tokens
        if matched:
            self.console.print("[green][OK] idempotent[/] (two identical compress() calls produced identical output)")
        else:
            logger.warning(
                "idempotence pre-check FAILED — compressor returned different outputs for identical input",
                extra={
                    "a_len": len(a.output),
                    "b_len": len(b.output),
                    "a_output_tokens": a.output_tokens,
                    "b_output_tokens": b.output_tokens,
                },
            )
            self.console.print("[yellow][WARN] idempotence drift detected[/] — see structured logs")
        return {
            "executed": True,
            "matched": matched,
            "a_output_tokens": a.output_tokens,
            "b_output_tokens": b.output_tokens,
        }

    def _print_task_summary(self, result: TaskResult) -> None:
        if not result.sweep_results:
            return
        primary = result.summary_stats.get("primary_metric", "f1")
        baseline_score = result.baseline_metrics.get(primary, {}).get("mean", 0.0)

        table = Table(title=f"{result.task_name} (baseline {primary}={baseline_score:.3f})")
        table.add_column("model")
        table.add_column("aggr", justify="right")
        table.add_column(primary, justify="right")
        table.add_column("delta vs base", justify="right")
        table.add_column("comp ratio", justify="right")
        table.add_column("ttc cost ($)", justify="right")
        table.add_column("p95 E2E (ms)", justify="right")
        table.add_column("fail %", justify="right")

        for cell in result.sweep_results:
            score = cell.metrics.get(primary, {}).get("mean", 0.0)
            delta = score - baseline_score
            delta_str = f"{delta:+.3f}"
            color = "green" if delta >= 0 else "red"
            table.add_row(
                cell.model,
                f"{cell.aggressiveness:.2f}",
                f"{score:.3f}",
                f"[{color}]{delta_str}[/]",
                f"{cell.avg_compression_ratio:.2%}",
                f"{cell.avg_ttc_cost_usd:.5f}",
                f"{cell.p95_e2e_latency_ms:.0f}",
                f"{cell.failure_rate:.0%}",
            )
        self.console.print(table)
        if result.best_cell:
            self.console.print(
                f"[bold green]-> sweet spot:[/] {result.best_cell.model} @ "
                f"aggressiveness={result.best_cell.aggressiveness:.2f}"
            )

    def _print_regression_summary(self, regression: RegressionReport) -> None:
        self.console.rule("[bold]regression check[/]")
        if not regression.regressions and not regression.improvements:
            self.console.print("[green]No regressions or improvements above threshold.[/]")
            return
        if regression.regressions:
            self.console.print(f"[red]{len(regression.regressions)} regressions[/] above threshold:")
            for d in regression.regressions[:10]:
                self.console.print(
                    f"  [red][FAIL][/] {d.task} {d.model}@{d.aggressiveness:.2f} "
                    f"{d.metric}: {d.baseline_value:.3f} -> {d.current_value:.3f} "
                    f"({d.delta:+.3f})"
                )
        if regression.improvements:
            self.console.print(f"[green]{len(regression.improvements)} improvements[/]:")
            for d in regression.improvements[:5]:
                self.console.print(
                    f"  [green][OK][/] {d.task} {d.model}@{d.aggressiveness:.2f} "
                    f"{d.metric}: {d.baseline_value:.3f} -> {d.current_value:.3f}"
                )
