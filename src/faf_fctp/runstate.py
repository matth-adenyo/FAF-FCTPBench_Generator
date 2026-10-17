"""Atomic checkpointing and human-readable pipeline progress."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary, path)
    return path


class ProgressReporter:
    def __init__(self, status_path: str | Path) -> None:
        self.status_path = Path(status_path)
        self.pipeline_started = time.perf_counter()
        self.stage_started = self.pipeline_started
        self.stage = "initializing"
        self.total = 0

    def begin(self, stage: str, total: int, *, message: str = "") -> None:
        self.stage = stage
        self.total = int(total)
        self.stage_started = time.perf_counter()
        self.update(0, current=None, message=message)

    def update(
        self,
        completed: int,
        *,
        current: str | None = None,
        message: str = "",
        extra: dict[str, object] | None = None,
    ) -> None:
        completed = int(completed)
        elapsed = max(0.0, time.perf_counter() - self.stage_started)
        percent = (100.0 * completed / self.total) if self.total else 100.0
        eta = None
        if completed > 0 and self.total > completed:
            eta = elapsed / completed * (self.total - completed)
        payload: dict[str, object] = {
            "status": "running",
            "stage": self.stage,
            "completed": completed,
            "total": self.total,
            "percent": percent,
            "current": current,
            "message": message,
            "stage_elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "pipeline_elapsed_seconds": max(0.0, time.perf_counter() - self.pipeline_started),
            "updated_at_utc": utc_now(),
        }
        if extra:
            payload.update(extra)
        atomic_write_json(self.status_path, payload)
        eta_text = "" if eta is None else f" ETA={eta/60:.1f}m"
        current_text = "" if current is None else f" current={current}"
        message_text = "" if not message else f" {message}"
        print(f"[{self.stage}] {completed}/{self.total} ({percent:5.1f}%){current_text}{eta_text}{message_text}", flush=True)

    def complete(self, *, message: str = "complete", extra: dict[str, object] | None = None) -> None:
        self.update(self.total, current=None, message=message, extra=extra)

    def pipeline_complete(self, outputs: dict[str, object]) -> None:
        payload = {
            "status": "complete",
            "stage": "complete",
            "completed": 1,
            "total": 1,
            "percent": 100.0,
            "current": None,
            "message": "pipeline complete",
            "pipeline_elapsed_seconds": max(0.0, time.perf_counter() - self.pipeline_started),
            "updated_at_utc": utc_now(),
            "outputs": outputs,
        }
        atomic_write_json(self.status_path, payload)
        print("[complete] pipeline complete", flush=True)

    def fail(self, error: BaseException) -> None:
        payload = {
            "status": "failed",
            "stage": self.stage,
            "completed": None,
            "total": self.total,
            "percent": None,
            "current": None,
            "message": str(error),
            "pipeline_elapsed_seconds": max(0.0, time.perf_counter() - self.pipeline_started),
            "updated_at_utc": utc_now(),
        }
        atomic_write_json(self.status_path, payload)
