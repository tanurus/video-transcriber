"""Background work: transcription (one GPU queue), AI finishing and Supabase sync.

Three independent workers so a slow network call never blocks the GPU queue:

* transcription — ``ThreadPoolExecutor(max_workers=1)``; queued jobs survive a
  restart (requeued from the database at boot);
* AI finishing — its own single worker, requeued from ``ai_status='queued'``;
* Supabase sync — an outbox loop over ``sync_status='pending'`` with
  exponential backoff, so rows arrive even after hours of network trouble.

Nothing here may raise into a discarded Future: every failure is logged and
recorded on the job.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from app import options as O

from .settings import WebSettings
from .storage import Job, Storage, utcnow_iso

MAX_TRACKED_JOBS = 200
BACKEND_WAIT_SECONDS = 1800  # how long a job waits for a down GPU server before failing
SYNC_POLL_SECONDS = 15


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(
        self,
        storage: Storage,
        settings: WebSettings,
        cfg_loader: Callable[[], object],
        transcribe_fn: Callable[..., Any],
        connections: Any = None,
        ai_factory: Optional[Callable[[Any], Any]] = None,
        sync_fn: Optional[Callable[..., None]] = None,
        backend_wait: int = BACKEND_WAIT_SECONDS,
        rnnoise_model: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.settings = settings
        self.cfg_loader = cfg_loader
        self.transcribe_fn = transcribe_fn
        self.connections = connections
        self.ai_factory = ai_factory
        self.sync_fn = sync_fn
        self.backend_wait = backend_wait
        self.rnnoise_model = rnnoise_model
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe")
        self._ai_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai")
        self._logs: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._sync_wake = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None

    # --- paths ----------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir

    def _dir(self, name: str) -> Path:
        d = self.data_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_path(self, job_id: str) -> Path:
        return self._dir("logs") / f"{job_id}.log"

    def ai_meta_path(self, job_id: str) -> Path:
        return self._dir("ai") / f"{job_id}.json"

    def kept_audio_path(self, root_id: str) -> Path:
        # Outside uploads/, which the retention sweep deletes wholesale.
        return self._dir("audio") / f"{root_id}.flac"

    def upload_file(self, job_id: str) -> Optional[Path]:
        job_dir = self.settings.uploads_dir / job_id
        if not job_dir.is_dir():
            return None
        files = sorted(p for p in job_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        return files[0] if files else None

    def read_text(self, rel: Optional[str]) -> Optional[str]:
        if not rel:
            return None
        path = (self.data_dir / rel).resolve()
        if not path.is_relative_to(self.data_dir.resolve()) or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def read_segments(self, job: Job) -> Optional[list]:
        raw = self.read_text(job.segments_path)
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    def read_ai_meta(self, job_id: str) -> dict:
        try:
            return json.loads(self.ai_meta_path(job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # --- logs -------------------------------------------------------------------------

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            if job_id not in self._logs and len(self._logs) >= MAX_TRACKED_JOBS:
                # Evict the oldest buffer to bound memory; the file copy remains.
                del self._logs[next(iter(self._logs))]
            self._logs.setdefault(job_id, []).append(line)
        try:
            with self.log_path(job_id).open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def get_logs(self, job_id: str, since: int = 0) -> tuple[list[str], int]:
        with self._lock:
            lines = self._logs.get(job_id)
            if lines is not None:
                return list(lines[since:]), len(lines)
        try:
            lines = self.log_path(job_id).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        return lines[since:], len(lines)

    # --- transcription ------------------------------------------------------------------

    def submit(self, job_id: str, video_path: Optional[Path] = None) -> None:
        self._executor.submit(self._run, job_id, video_path)

    def _wait_for_backend(self, job_id: str):
        """Load the backend config, waiting out a temporarily down GPU server."""
        deadline = time.monotonic() + self.backend_wait
        announced = False
        while True:
            try:
                return self.cfg_loader()
            except RuntimeError as e:
                if "not responding" not in str(e) or time.monotonic() >= deadline:
                    raise
                if not announced:
                    self._append_log(job_id, f"Waiting for the GPU server: {e}")
                    announced = True
                time.sleep(15)

    def _run(self, job_id: str, video_path: Optional[Path] = None) -> None:
        # Nothing may escape this method: the executor discards the Future, so an
        # uncaught exception would silently leave the job stuck forever.
        try:
            self.storage.update_status(job_id, "running")
            job = self.storage.get_job(job_id)
            if job is None:
                raise RuntimeError("job disappeared from the database")
            source = Path(video_path) if video_path else self.upload_file(job_id)
            self._append_log(job_id, f"--- {job.original_name} ---")
            opts = O.resolve(job.settings)
            cfg = self._wait_for_backend(job_id)
            root = self.storage.root_of(job_id)
            kept = Path(job.audio_path) if job.audio_path else self.kept_audio_path(root)
            if not kept.exists() and (source is None or not source.exists()):
                raise RuntimeError("the original upload and the kept audio are both gone; cannot transcribe")
            result = self.transcribe_fn(
                source, cfg, opts, kept,
                logger=lambda m: self._append_log(job_id, m),
                rnnoise_model=self.rnnoise_model,
            )
            self._store_result(job_id, result, kept)
            # Final log line before the terminal status: a poll that sees the
            # terminal status must already see the full log.
            self._append_log(job_id, "Done.")
            self.storage.update_status(job_id, "done", completed_at=_utcnow_iso())
            self._after_transcription(job_id, opts)
        except Exception as e:  # noqa: BLE001 - surface any failure as job error
            self._append_log(job_id, f"Error: {e}")
            try:
                self.storage.update_status(job_id, "error", completed_at=_utcnow_iso(), error=str(e))
            except Exception:  # noqa: BLE001
                # Storage itself is failing; the log line above is the only signal left.
                pass

    def _store_result(self, job_id: str, result: Any, kept: Path) -> None:
        tdir = self.settings.transcripts_dir
        tdir.mkdir(parents=True, exist_ok=True)
        out = tdir / f"{job_id}.txt"
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(result.text, encoding="utf-8")
        tmp.replace(out)  # atomic: never a truncated transcript at the final path
        seg_path = self._dir("segments") / f"{job_id}.json"
        seg_path.write_text(json.dumps(result.segments, ensure_ascii=False), encoding="utf-8")
        self.storage.update_job(
            job_id,
            transcript_path=str(out.relative_to(self.data_dir)).replace("\\", "/"),
            segments_path=str(seg_path.relative_to(self.data_dir)).replace("\\", "/"),
            duration_sec=result.duration,
            languages=",".join(result.languages),
            provider=result.provider,
            model=result.model,
            audio_path=str(kept),
            error=None,
        )

    def _after_transcription(self, job_id: str, opts: Dict[str, Any]) -> None:
        conns = self.connections.load() if self.connections else None
        if opts.get("ai_finish") and conns is not None and conns.ai_ready:
            self.enqueue_ai([job_id])
        elif opts.get("ai_finish"):
            self._append_log(job_id, "AI finishing skipped: add an OpenAI key in Settings, then run it from the library.")
            if opts.get("sync_supabase"):
                self.mark_sync([job_id])
        elif opts.get("sync_supabase"):
            self.mark_sync([job_id])

    # --- regenerate ---------------------------------------------------------------------

    def regenerate(self, job_id: str, raw: Dict[str, Any], preset: Optional[str] = None,
                   keep_base: bool = True) -> str:
        parent = self.storage.get_job(job_id)
        if parent is None:
            raise KeyError(job_id)
        family = self.storage.versions_of(job_id)
        opts = O.resolve(raw, base=parent.settings if keep_base else None, preset=preset)
        opts["_summary"] = O.summary(opts)
        new_id = uuid.uuid4().hex
        root = self.storage.root_of(job_id)
        kept = Path(parent.audio_path) if parent.audio_path else self.kept_audio_path(root)
        source = self.upload_file(root) or self.upload_file(job_id)
        if not kept.exists() and source is None:
            raise RuntimeError("Neither the kept audio nor the original upload exists any more.")
        self.storage.create_job(
            new_id, parent.original_name, utcnow_iso(),
            parent_id=job_id,
            version=max((j.version or 1) for j in family) + 1,
            source=parent.source, source_detail=parent.source_detail,
            recorded_at=parent.recorded_at, file_size=parent.file_size, file_sha256=parent.file_sha256,
            settings_json=json.dumps(opts), audio_path=str(kept) if kept.exists() else None,
        )
        self._append_log(new_id, f"Regenerating {parent.display_name} (v{parent.version or 1}) with: {opts['_summary']}")
        self.submit(new_id, source)
        return new_id

    # --- AI finishing ---------------------------------------------------------------------

    def enqueue_ai(self, job_ids: Iterable[str]) -> int:
        count = 0
        for job_id in job_ids:
            job = self.storage.get_job(job_id)
            if job is None or job.status != "done":
                continue
            self.storage.update_job(job_id, ai_status="queued", ai_error=None)
            self._ai_executor.submit(self._ai_run, job_id)
            count += 1
        return count

    def _ai_run(self, job_id: str) -> None:
        try:
            job = self.storage.get_job(job_id)
            if job is None or job.ai_status not in ("queued", "running"):
                return
            conns = self.connections.load()
            if not conns.ai_ready:
                raise RuntimeError("No OpenAI key configured (Settings → Connections).")
            text = self.read_text(job.transcript_path)
            if not text:
                raise RuntimeError("The transcript is empty — nothing to clean.")
            self.storage.update_job(job_id, ai_status="running")
            finisher = self.ai_factory(conns)
            result = finisher.finish(text, log=lambda m: self._append_log(job_id, m))
            clean_path = self._dir("clean") / f"{job_id}.txt"
            clean_path.write_text(result.clean_text, encoding="utf-8")
            self.ai_meta_path(job_id).write_text(json.dumps({
                "model": conns.openai_model, "language": result.language, "people": result.people,
                "topics": result.topics, "notes": result.notes, "at": _utcnow_iso(),
                "rejected_chunks": result.rejected_chunks, "kept_chunks": result.kept_chunks,
            }, ensure_ascii=False), encoding="utf-8")
            self.storage.update_job(
                job_id, ai_status="done", ai_error=None, ai_at=_utcnow_iso(),
                title=result.title, description=result.description, tags=json.dumps(result.tags, ensure_ascii=False),
                clean_text_path=str(clean_path.relative_to(self.data_dir)).replace("\\", "/"),
            )
            self._append_log(job_id, f"AI finishing done: \"{result.title}\"")
            opts = O.resolve(job.settings)
            if opts.get("sync_supabase") or job.sync_status in ("synced", "pending", "error"):
                self.mark_sync([job_id])  # the row must carry the new title/description
        except Exception as e:  # noqa: BLE001
            self._append_log(job_id, f"AI finishing failed: {e}")
            try:
                self.storage.update_job(job_id, ai_status="error", ai_error=str(e)[:500])
            except Exception:  # noqa: BLE001
                pass

    # --- Supabase sync ---------------------------------------------------------------------

    def mark_sync(self, job_ids: Iterable[str]) -> int:
        count = 0
        for job_id in job_ids:
            job = self.storage.get_job(job_id)
            if job is None or job.status != "done":
                continue
            self.storage.update_job(job_id, sync_status="pending", sync_next_at=None, sync_error=None,
                                    sync_attempts=0)
            count += 1
        self._sync_wake.set()
        return count

    def build_row(self, job: Job) -> Dict[str, Any]:
        from .supabase_sync import build_row
        return build_row(
            job,
            raw_text=self.read_text(job.transcript_path) or "",
            clean_text=self.read_text(job.clean_text_path),
            segments=self.read_segments(job),
            root_id=self.storage.root_of(job.id),
            ai_meta=self.read_ai_meta(job.id),
        )

    def sync_once(self) -> int:
        """Deliver every due pending job. Returns how many were synced."""
        from .supabase_sync import SupabaseConfig, next_attempt_iso
        due = self.storage.due_for_sync(_utcnow_iso())
        if not due:
            return 0
        conns = self.connections.load() if self.connections else None
        if conns is None or not conns.supabase_ready:
            for job in due:
                if job.sync_error != "Waiting for Supabase settings":
                    self.storage.update_job(job.id, sync_error="Waiting for Supabase settings")
            return 0
        cfg = SupabaseConfig(conns.supabase_url, conns.supabase_key, conns.supabase_table)
        synced = 0
        for job in due:
            try:
                self.sync_fn(cfg, [self.build_row(job)])
            except Exception as e:  # noqa: BLE001
                attempts = (job.sync_attempts or 0) + 1
                self.storage.update_job(job.id, sync_error=str(e)[:500], sync_attempts=attempts,
                                        sync_next_at=next_attempt_iso(attempts))
                self._append_log(job.id, f"Supabase sync failed (attempt {attempts}): {e}")
                continue
            self.storage.update_job(job.id, sync_status="synced", synced_at=_utcnow_iso(), sync_error=None,
                                    sync_attempts=0, sync_next_at=None)
            self._append_log(job.id, "Sent to Supabase.")
            synced += 1
        return synced

    def start_sync_loop(self) -> None:
        if self._sync_thread is not None:
            return

        def loop() -> None:
            while True:
                try:
                    self.sync_once()
                except Exception:  # noqa: BLE001 - the loop must never die
                    pass
                self._sync_wake.wait(SYNC_POLL_SECONDS)
                self._sync_wake.clear()

        self._sync_thread = threading.Thread(target=loop, name="supabase-sync", daemon=True)
        self._sync_thread.start()

    # --- boot recovery / deletion ------------------------------------------------------------

    def resume(self) -> None:
        """Requeue work that was pending when the process stopped."""
        for job in self.storage.jobs_with(status="queued"):
            source = self.upload_file(job.id)
            kept = Path(job.audio_path) if job.audio_path else self.kept_audio_path(self.storage.root_of(job.id))
            if source or kept.exists():
                self.submit(job.id, source)
            else:
                self.storage.update_status(job.id, "interrupted", error="upload missing after restart")
        queued_ai = [j.id for j in self.storage.jobs_with(ai_status="queued")]
        for job_id in queued_ai:
            self._ai_executor.submit(self._ai_run, job_id)

    def delete(self, job_ids: Iterable[str]) -> int:
        count = 0
        for job_id in job_ids:
            job = self.storage.get_job(job_id)
            if job is None or job.status in ("queued", "running"):
                continue
            root = self.storage.root_of(job_id)
            family = [j for j in self.storage.versions_of(job_id) if j.id != job_id]
            self.storage.delete_job(job_id)
            for rel in (job.transcript_path, job.segments_path, job.clean_text_path):
                if rel:
                    (self.data_dir / rel).unlink(missing_ok=True)
            self.ai_meta_path(job_id).unlink(missing_ok=True)
            self.log_path(job_id).unlink(missing_ok=True)
            if not family:  # last version of this recording: drop its audio and upload too
                shutil.rmtree(self.settings.uploads_dir / job_id, ignore_errors=True)
                shutil.rmtree(self.settings.uploads_dir / root, ignore_errors=True)
                if job.audio_path:
                    Path(job.audio_path).unlink(missing_ok=True)
            count += 1
        return count


def purge_old_videos(
    settings: WebSettings, now: Optional[float] = None, storage: Optional[Storage] = None
) -> int:
    """Delete upload directories whose mtime is older than the retention window.

    Directories belonging to jobs that are still queued or running are kept
    regardless of age. One unreadable entry never aborts the sweep. Returns the
    number of directories removed. Transcripts and the kept lossless audio
    (data/audio/) are never touched, so regeneration keeps working.
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


def purge_stale_incoming(settings: WebSettings, now: Optional[float] = None, max_age_hours: int = 48) -> int:
    """Drop chunked uploads that were started but never completed."""
    now = now or time.time()
    incoming = settings.data_dir / "incoming"
    if not incoming.exists():
        return 0
    removed = 0
    for child in incoming.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < now - max_age_hours * 3600:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
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
                purge_stale_incoming(settings)
            except Exception:  # noqa: BLE001 - cleanup must never crash the app
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, name="video-cleanup", daemon=True)
    thread.start()
    return thread
