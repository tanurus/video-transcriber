from __future__ import annotations

import json
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

# Per-job choices made on the upload form live beside the upload, so a job that
# is requeued after a restart still runs with them.
OPTIONS_FILE = ".options.json"
QUALITIES = ("best", "fast")
# Both the local GPU server and Groq serve the turbo model under this id.
FAST_MODEL = "whisper-large-v3-turbo"


def write_options(job_dir: Path, options: dict) -> None:
    (Path(job_dir) / OPTIONS_FILE).write_text(json.dumps(options), encoding="utf-8")


def read_options(video_path: Path) -> dict:
    try:
        return json.loads((Path(video_path).parent / OPTIONS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def apply_options(cfg, options: dict) -> Optional[str]:
    """Apply upload-form options to a loaded Config. Returns a log line, or None."""
    if options.get("quality") != "fast":
        return None
    if getattr(cfg, "provider", None) in ("local", "groq"):
        cfg.model = FAST_MODEL
        return f"Quality: fast ({FAST_MODEL})"
    return f"Quality: fast is not available on {getattr(cfg, 'provider', '?')}; using {cfg.model}"


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
        # Nothing may escape this method: the executor discards the Future, so
        # an uncaught exception would silently leave the job stuck forever.
        out_path = self.settings.transcripts_dir / f"{job_id}.txt"
        try:
            self.storage.update_status(job_id, "running")
            self._append_log(job_id, f"--- {Path(video_path).name} ---")
            cfg = self.cfg_loader()
            note = apply_options(cfg, read_options(video_path))
            if note:
                self._append_log(job_id, note)
            self.transcribe_fn(
                video_path,
                cfg,
                logger=lambda m: self._append_log(job_id, m),
                out_path=out_path,
            )
            rel = out_path.relative_to(self.settings.data_dir)
            # Final log line before the terminal status: a poll that sees the
            # terminal status must already see the full log.
            self._append_log(job_id, "Done.")
            self.storage.update_status(
                job_id, "done", completed_at=_utcnow_iso(), transcript_path=str(rel).replace("\\", "/")
            )
        except Exception as e:  # noqa: BLE001 - surface any failure as job error
            self._append_log(job_id, f"Error: {e}")
            try:
                self.storage.update_status(job_id, "error", completed_at=_utcnow_iso(), error=str(e))
            except Exception:  # noqa: BLE001
                # Storage itself is failing; the log line above is the only signal left.
                pass


def purge_old_videos(
    settings: WebSettings, now: Optional[float] = None, storage: Optional[Storage] = None
) -> int:
    """Delete upload directories whose mtime is older than the retention window.

    Directories belonging to jobs that are still queued or running are kept
    regardless of age. One unreadable entry never aborts the sweep. Returns the
    number of directories removed. Transcripts are never touched.
    """
    if now is None:
        now = time.time()
    uploads = settings.uploads_dir
    if not uploads.exists():
        return 0
    cutoff = now - settings.retain_video_days * 86400
    removed = 0
    for child in uploads.iterdir():
        try:
            if not child.is_dir() or child.stat().st_mtime >= cutoff:
                continue
            if storage is not None:
                job = storage.get_job(child.name)
                if job is not None and job.status in ("queued", "running"):
                    continue
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        except Exception:  # noqa: BLE001 - skip the bad entry, keep sweeping
            continue
    return removed


def start_cleanup_thread(
    settings: WebSettings, storage: Optional[Storage] = None, interval_seconds: int = 86400
) -> threading.Thread:
    """Run an immediate purge, then purge every `interval_seconds`, in a daemon thread."""

    def _loop() -> None:
        while True:
            try:
                purge_old_videos(settings, storage=storage)
            except Exception:  # noqa: BLE001 - cleanup must never crash the app
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="video-cleanup", daemon=True)
    thread.start()
    return thread
