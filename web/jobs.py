from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .settings import WebSettings
from .storage import Storage


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(
        self,
        storage: Storage,
        settings: WebSettings,
        cfg_loader: Callable[[], object],
        transcribe_fn: Callable[..., Path],
    ) -> None:
        self.storage = storage
        self.settings = settings
        self.cfg_loader = cfg_loader
        self.transcribe_fn = transcribe_fn
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._logs: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def submit(self, job_id: str, video_path: Path) -> None:
        self._executor.submit(self._run, job_id, video_path)

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            self._logs.setdefault(job_id, []).append(line)

    def get_logs(self, job_id: str, since: int = 0) -> tuple[list[str], int]:
        with self._lock:
            lines = self._logs.get(job_id, [])
            return list(lines[since:]), len(lines)

    def _run(self, job_id: str, video_path: Path) -> None:
        self.storage.update_status(job_id, "running")
        self._append_log(job_id, f"--- {Path(video_path).name} ---")
        out_path = self.settings.transcripts_dir / f"{job_id}.txt"
        try:
            cfg = self.cfg_loader()
            self.transcribe_fn(
                video_path,
                cfg,
                logger=lambda m: self._append_log(job_id, m),
                out_path=out_path,
            )
            rel = out_path.relative_to(self.settings.data_dir)
            self.storage.update_status(
                job_id, "done", completed_at=_utcnow_iso(), transcript_path=str(rel).replace("\\", "/")
            )
            self._append_log(job_id, "Done.")
        except Exception as e:  # noqa: BLE001 - surface any failure as job error
            self.storage.update_status(job_id, "error", completed_at=_utcnow_iso(), error=str(e))
            self._append_log(job_id, f"Error: {e}")
