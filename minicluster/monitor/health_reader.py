"""Health checkpoint reader for live MiniCluster monitoring."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from minicluster.runner.distributed_runner import HealthCheckpoint, WorkerSnapshot


class HealthReader:
    """Read and watch HealthCheckpoint files emitted during training."""

    def __init__(
        self,
        checkpoint_path: str,
        timeout_seconds: float = 30.0,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.timeout_seconds = timeout_seconds
        self.stop_event = stop_event or threading.Event()
        self._latest: Optional[HealthCheckpoint] = None
        self._start_time = time.time()

    def read(self) -> Optional[HealthCheckpoint]:
        """Read current checkpoint file, returning None when unavailable."""
        path = Path(self.checkpoint_path)
        if not path.exists():
            time.sleep(0.05)
            if not path.exists():
                return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        workers = [
            WorkerSnapshot(**worker) for worker in payload.get("workers", [])
        ]
        checkpoint = HealthCheckpoint(
            run_id=str(payload.get("run_id", "")),
            total_steps=int(payload.get("total_steps", 0)),
            completed_steps=int(payload.get("completed_steps", 0)),
            workers=workers,
            timestamp=str(payload.get("timestamp", "")),
            is_complete=bool(payload.get("is_complete", False)),
        )
        self._latest = checkpoint
        return checkpoint

    def watch(
        self,
        callback: Callable[[HealthCheckpoint], None],
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Poll checkpoint path and invoke callback on step progress changes."""
        last_completed = -1
        while not self.stop_event.is_set():
            checkpoint = self.read()
            if checkpoint is None:
                if self.is_run_complete():
                    return
                time.sleep(poll_interval_seconds)
                continue

            if checkpoint.completed_steps != last_completed:
                callback(checkpoint)
                last_completed = checkpoint.completed_steps

            if checkpoint.is_complete:
                return
            time.sleep(poll_interval_seconds)

    def is_run_complete(self) -> bool:
        """Return True when run is complete, or when checkpoint times out."""
        if self._latest is not None and self._latest.is_complete:
            return True
        if Path(self.checkpoint_path).exists():
            return False
        return (time.time() - self._start_time) >= self.timeout_seconds

