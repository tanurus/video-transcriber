from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .settings import WebSettings
from .storage import Storage

MAX_TRACKED_JOBS = 50


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
            if job_id not in self._logs and len(self._logs) >= MAX_TRACKED_JOBS:
                # Evict the oldest tracked job's buffer to bound memory on a long-lived server.
                oldest = next(iter(self._logs))
                del self._logs[oldest]
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


def purge_old_videos(settings: WebSettings, now: Optional[float] = None) -> int:
    """Delete upload directories whose mtime is older than the retention window.

    Returns the number of directories removed. Transcripts are never touched.
    """
    if now is None:
        now = time.time()
    uploads = settings.uploads_dir
    if not uploads.exists():
        return 0
    cutoff = now - settings.retain_video_days * 86400
    removed = 0
    for child in uploads.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def start_cleanup_thread(settings: WebSettings, interval_seconds: int = 86400) -> threading.Thread:
    """Run an immediate purge, then purge every `interval_seconds`, in a daemon thread."""

    def _loop() -> None:
        while True:
            try:
                purge_old_videos(settings)
            except Exception:  # noqa: BLE001 - cleanup must never crash the app
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="video-cleanup", daemon=True)
    thread.start()
    return thread
