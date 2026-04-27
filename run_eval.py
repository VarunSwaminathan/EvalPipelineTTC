"""CLI entrypoint. Wraps argparse + EvalRunner + report.py.

Smoke-test contract:

    python run_eval.py --llm mock --samples 5

must produce a complete HTML report in ``results/`` with no API keys set.
The CLI auto-falls-back to MockTTCClient when ``--llm mock`` is selected
and ``TTC_API_KEY`` is missing.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np

from eval.client import MockTTCClient, TTCClient
from eval.llm_backend import build_backend
from eval.logging_config import setup_logging
from eval.runner import EvalRunner
from eval.tasks.base import EvalTask
from eval.tasks.conversational import ConversationalQATask
from eval.tasks.long_context import LongContextNeedleTask
from eval.tasks.qa import QATask
from eval.tasks.rag import RAGTask
from eval.tasks.summarization import SummarizationTask

logger = logging.getLogger(__name__)

TASK_REGISTRY = {
    "qa": QATask,
    "summarization": SummarizationTask,
    "rag": RAGTask,
    "conversational": ConversationalQATask,
    "long_context": LongContextNeedleTask,
}

DEFAULT_TASKS = list(TASK_REGISTRY.keys())
DEFAULT_MODELS = ["bear-1", "bear-1.1", "bear-1.2"]
DEFAULT_AGGRESSIVENESS = [0.1, 0.3, 0.5, 0.7, 0.9]

def _csv(s: str) -> list[str]:
    return [item.strip() for item in s.split(",") if item.strip()]

def _csv_float(s: str) -> list[float]:
    return [float(x) for x in _csv(s)]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Run the TTC eval pipeline across (task × model × aggressiveness).",
    )
    p.add_argument(
        "--tasks",
        type=_csv,
        default=DEFAULT_TASKS,
        help=f"Comma-separated subset of tasks. Default: {','.join(DEFAULT_TASKS)}",
    )
    p.add_argument(
        "--models",
        type=_csv,
        default=DEFAULT_MODELS,
        help=f"TTC models to sweep. Default: {','.join(DEFAULT_MODELS)}",
    )
    p.add_argument(
        "--aggressiveness",
        type=_csv_float,
        default=DEFAULT_AGGRESSIVENESS,
        help=f"Aggressiveness levels. Default: {','.join(map(str, DEFAULT_AGGRESSIVENESS))}",
    )
    p.add_argument(
        "--llm",
        choices=["mock", "openai"],
        default="mock",
        help="Downstream LLM backend (default: mock — no keys required)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Max samples per task (each task may have fewer in its dataset).",
    )
    p.add_argument("--workers", type=int, default=4, help="ThreadPool workers per task.")
    p.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Enable sentence-transformers + NLI faithfulness (slow, accurate).",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to a prior results.json for regression diff.",
    )
    p.add_argument("--output", type=str, default="results", help="Output directory.")
    p.add_argument("--no-report", action="store_true", help="Skip HTML report generation.")
    p.add_argument("--log-json", action="store_true", help="Emit JSON-line structured logs.")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, …)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument(
        "--ttc-rpm",
        type=int,
        default=50,
        help=(
            "Client-side TTC rate limit (requests/minute). Default 50 paces "
            "under the documented 60/min cap with 10/min headroom. Set to 0 "
            "to disable pacing if you have a higher-tier quota."
        ),
    )
    return p.parse_args(argv)

def _build_tasks(names: list[str], use_embeddings: bool) -> list[EvalTask]:
    out: list[EvalTask] = []
    for name in names:
        if name not in TASK_REGISTRY:
            raise SystemExit(f"unknown task: {name!r}. Valid: {','.join(TASK_REGISTRY)}")
        cls = TASK_REGISTRY[name]
        if name in ("summarization", "rag"):
            out.append(cls(use_embeddings=use_embeddings))  # type: ignore[call-arg]
        else:
            out.append(cls())
    return out

def _build_ttc_client(llm_choice: str, requests_per_minute: int | None = None):  # type: ignore[no-untyped-def]
    """If we're running the smoke test (mock LLM, no key), use the mock client."""
    has_key = bool(os.environ.get("TTC_API_KEY"))
    if not has_key:
        if llm_choice == "mock":
            logger.warning(
                "TTC_API_KEY not set; using MockTTCClient. "
                "Compression will not match the real bear-1.x models. "
                "Use this only for smoke-testing the pipeline."
            )
            return MockTTCClient()
        raise SystemExit(
            "TTC_API_KEY is not set. Set it in your environment, or use "
            "--llm mock to run the no-keys smoke test."
        )
    rpm = requests_per_minute if requests_per_minute else None
    return TTCClient(requests_per_minute=rpm)

def _load_dotenv_if_available() -> None:
    """Load OPENAI_API_KEY / TTC_API_KEY from ``.env`` if python-dotenv is installed.

    Soft dependency: if ``python-dotenv`` isn't installed, the user can still
    ``export`` the vars manually. We don't make the package mandatory.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)

def main(argv: list[str] | None = None) -> int:
    import contextlib
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")

    _load_dotenv_if_available()

    args = parse_args(argv)
    setup_logging(level=args.log_level, json_output=args.log_json)

    random.seed(args.seed)
    np.random.seed(args.seed)

    client = _build_ttc_client(args.llm, requests_per_minute=args.ttc_rpm)
    backend = build_backend(args.llm)
    tasks = _build_tasks(args.tasks, use_embeddings=args.use_embeddings)

    runner = EvalRunner(
        client=client,
        llm=backend,
        tasks=tasks,
        models=args.models,
        aggressiveness_levels=args.aggressiveness,
        max_workers=args.workers,
        seed=args.seed,
        use_embeddings=args.use_embeddings,
        output_dir=args.output,
        baseline_path=args.baseline,
    )
    report, json_path = runner.run(max_samples=args.samples)

    if not args.no_report:
        from report import render_report

        html_path = json_path.with_suffix(".html")
        render_report(report_dict=_to_dict(report), out_path=html_path)
        runner.console.print(f"[bold]Report:[/] {html_path}")

    return 0

def _to_dict(report) -> dict:  # type: ignore[no-untyped-def]
    from dataclasses import asdict

    return asdict(report)

if __name__ == "__main__":
    sys.exit(main())
