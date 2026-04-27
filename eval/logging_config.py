"""Structured logging for the eval pipeline.

Two modes:

* **Plain text** (default): readable for terminal use during a run.
* **JSON-line** (`--log-json`): one JSON object per line, suitable for piping
  into ``jq`` or shipping to a log aggregator.

We use stdlib ``logging`` deliberately — no extra dependency, and every module
just does ``logger = logging.getLogger(__name__)``. The TTC client logs a fixed
set of fields per request (model, aggressiveness, latency_ms, cost_usd,
attempt, …) so that production observability gets useful structured data
without bolting on a separate framework.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonLineFormatter(logging.Formatter):
    """One JSON object per log record. Stable field set for log aggregators."""

    _STD_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._STD_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class HumanFormatter(logging.Formatter):
    """Compact human-readable formatter. Appends ``extra=`` fields inline."""

    _STD_ATTRS = JsonLineFormatter._STD_ATTRS

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<5} "
            f"{record.name}: {record.getMessage()}"
        )
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._STD_ATTRS and not k.startswith("_")
        }
        if extras:
            extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
            base = f"{base} [{extras_str}]"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base

def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Idempotent logging setup. Safe to call from CLI entrypoint and from tests."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonLineFormatter() if json_output else HumanFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("urllib3", "openai._base_client", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
