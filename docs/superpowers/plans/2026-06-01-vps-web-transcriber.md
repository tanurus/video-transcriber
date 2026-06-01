# VPS Web Transcriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Dockerized Flask web front-end that lets the user upload one video through the browser (over their Tailscale tailnet), watch live transcription progress, and download the resulting `.txt` — reusing the existing transcription core.

**Architecture:** A new `web/` package wraps the existing `app/` core. Flask (served by gunicorn, single worker + threads) accepts an upload, saves it under a `/data` volume, and hands it to a single-slot background worker that calls the existing `transcribe_video()`. Job state lives in SQLite; the progress page polls a JSON API that streams the same log lines the desktop app shows. The container is published only to `127.0.0.1` and exposed to the tailnet via `tailscale serve`.

**Tech Stack:** Python 3.12, Flask, gunicorn, SQLite (stdlib), Docker + docker-compose, ffmpeg (in image), Groq Whisper API (existing). Tests: pytest.

**Spec:** `docs/superpowers/specs/2026-06-01-vps-web-transcriber-design.md`

---

## File Structure

**Created:**
- `web/__init__.py` — Flask app factory `create_app`; wires settings/storage/jobs, reconciles interrupted jobs, optionally starts cleanup thread.
- `web/settings.py` — `WebSettings` dataclass loaded from env (data dir, retention, max size, allowed extensions).
- `web/storage.py` — `Storage` class: SQLite schema + job CRUD; `Job` dataclass.
- `web/jobs.py` — `JobManager` (executor, per-job log buffers, run lifecycle) + `purge_old_videos` retention helper + cleanup thread starter.
- `web/routes.py` — `register_routes(app)`: all HTTP endpoints.
- `web/wsgi.py` — gunicorn entrypoint exposing `app`.
- `web/templates/{base,index,job,history}.html` — UI.
- `web/static/{style.css,job.js}` — minimal styling + progress poller.
- `Dockerfile`, `.dockerignore`, `docker-compose.yml` — container runtime.
- `requirements-web.txt` — Flask + gunicorn (image deps).
- `requirements-dev.txt` — pytest (local test deps).
- `deploy.sh` — VPS deploy script.
- `docs/deploy.md` — one-time setup + redeploy instructions.
- `tests/conftest.py`, `tests/test_*.py` — tests.

**Modified:**
- `app/main.py` — add optional `out_path` parameter to `transcribe_video` (backward compatible).
- `README.md` — add a "Web app on a VPS" section.

---

## Task 0: Test tooling

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create dev requirements**

`requirements-dev.txt`:
```
pytest>=8.0
flask>=3.0
```
(Flask is needed locally to import the `web` package during tests; gunicorn is intentionally omitted — it only runs inside the Linux container.)

- [ ] **Step 2: Install dev + web deps into the venv**

Run: `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`
Expected: pytest and flask install successfully.

- [ ] **Step 3: Create an empty tests package and a smoke test**

`tests/__init__.py`: (empty file)

`tests/test_smoke.py`:
```python
def test_pytest_runs():
    assert True
```

- [ ] **Step 4: Confirm pytest runs**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt tests/__init__.py tests/test_smoke.py
git commit -m "test: add pytest tooling and smoke test"
```

---

## Task 1: Add `out_path` to `transcribe_video`

**Files:**
- Modify: `app/main.py` (the `transcribe_video` function)
- Test: `tests/test_transcribe_outpath.py`

- [ ] **Step 1: Write the failing test**

`tests/test_transcribe_outpath.py`:
```python
from pathlib import Path

import app.main as m
from app.config import Config


def test_transcribe_writes_to_out_path(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"y" * 10)  # small => no chunking

    # Fake the two external boundaries: ffmpeg extraction and the Groq client.
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f):
            return "hello world"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)

    cfg = Config(openai_api_key="x")
    out = tmp_path / "out" / "result.txt"

    result = m.transcribe_video(video, cfg, logger=lambda s: None, out_path=out)

    assert result == out
    assert out.read_text(encoding="utf-8") == "hello world"


def test_transcribe_defaults_next_to_video(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"y" * 10)
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f):
            return "abc"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)
    cfg = Config(openai_api_key="x")

    result = m.transcribe_video(video, cfg, logger=lambda s: None)

    assert result == video.with_suffix(".txt")
    assert result.read_text(encoding="utf-8") == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_transcribe_outpath.py -v`
Expected: FAIL — `transcribe_video() got an unexpected keyword argument 'out_path'`.

- [ ] **Step 3: Add the `out_path` parameter**

In `app/main.py`, change the `transcribe_video` signature:
```python
def transcribe_video(
    video_path: Path,
    cfg: Config,
    logger: Callable[[str], None] | None = None,
    use_tqdm: bool = False,
    out_path: Path | None = None,
) -> Path:
```

Then change the output-writing lines at the end of the function from:
```python
    out_txt = video_path.with_suffix(".txt")
    out_txt.write_text(transcript, encoding="utf-8")
    log(f"Transcript saved to {out_txt}")
    return out_txt
```
to:
```python
    out_txt = Path(out_path) if out_path else video_path.with_suffix(".txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(transcript, encoding="utf-8")
    log(f"Transcript saved to {out_txt}")
    return out_txt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_transcribe_outpath.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_transcribe_outpath.py
git commit -m "feat: optional out_path for transcribe_video (backward compatible)"
```

---

## Task 2: Web settings

**Files:**
- Create: `web/__init__.py` (temporary minimal version, replaced in Task 6)
- Create: `web/settings.py`
- Create: `tests/conftest.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Create a minimal package init (placeholder)**

`web/__init__.py`:
```python
# Replaced with the Flask app factory in Task 6.
```

- [ ] **Step 2: Write the failing test**

`tests/test_settings.py`:
```python
from pathlib import Path

from web.settings import WebSettings


def test_load_defaults(monkeypatch):
    for var in ["DATA_DIR", "RETAIN_VIDEO_DAYS", "MAX_CONTENT_MB", "ALLOWED_EXT"]:
        monkeypatch.delenv(var, raising=False)
    s = WebSettings.load()
    assert s.data_dir == Path("/data")
    assert s.retain_video_days == 30
    assert s.max_content_mb == 2048
    assert "mp4" in s.allowed_ext
    assert s.uploads_dir == Path("/data/uploads")
    assert s.transcripts_dir == Path("/data/transcripts")
    assert s.db_path == Path("/data/app.db")


def test_load_overrides(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/tmp/d")
    monkeypatch.setenv("RETAIN_VIDEO_DAYS", "7")
    monkeypatch.setenv("MAX_CONTENT_MB", "100")
    monkeypatch.setenv("ALLOWED_EXT", "mp4, .MKV ,mov")
    s = WebSettings.load()
    assert s.data_dir == Path("/tmp/d")
    assert s.retain_video_days == 7
    assert s.max_content_mb == 100
    assert s.allowed_ext == frozenset({"mp4", "mkv", "mov"})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.settings'`.

- [ ] **Step 4: Implement settings**

`web/settings.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOWED_EXT = "mp4,mkv,mov,avi,webm,m4a,mp3,wav"


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path
    retain_video_days: int
    max_content_mb: int
    allowed_ext: frozenset[str]

    @staticmethod
    def load() -> "WebSettings":
        raw_ext = os.getenv("ALLOWED_EXT", DEFAULT_ALLOWED_EXT)
        allowed = frozenset(
            e.strip().lower().lstrip(".") for e in raw_ext.split(",") if e.strip()
        )
        return WebSettings(
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            retain_video_days=int(os.getenv("RETAIN_VIDEO_DAYS", "30")),
            max_content_mb=int(os.getenv("MAX_CONTENT_MB", "2048")),
            allowed_ext=allowed,
        )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"
```

- [ ] **Step 5: Create the shared test fixture**

`tests/conftest.py`:
```python
import pytest

from web.settings import WebSettings


@pytest.fixture
def settings(tmp_path):
    s = WebSettings(
        data_dir=tmp_path,
        retain_video_days=30,
        max_content_mb=2048,
        allowed_ext=frozenset({"mp4", "mkv", "mov", "avi", "webm", "m4a", "mp3", "wav"}),
    )
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.transcripts_dir.mkdir(parents=True, exist_ok=True)
    return s
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_settings.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add web/__init__.py web/settings.py tests/conftest.py tests/test_settings.py
git commit -m "feat: web settings loaded from environment"
```

---

## Task 3: SQLite storage layer

**Files:**
- Create: `web/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_storage.py`:
```python
from web.storage import Storage, Job


def test_create_and_get(settings):
    store = Storage(settings.db_path)
    store.create_job("abc", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    job = store.get_job("abc")
    assert isinstance(job, Job)
    assert job.id == "abc"
    assert job.original_name == "Clip.mp4"
    assert job.status == "queued"
    assert job.completed_at is None


def test_get_missing_returns_none(settings):
    store = Storage(settings.db_path)
    assert store.get_job("nope") is None


def test_update_status_to_done(settings):
    store = Storage(settings.db_path)
    store.create_job("abc", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status(
        "abc", "done",
        completed_at="2026-06-01T00:05:00+00:00",
        transcript_path="transcripts/abc.txt",
    )
    job = store.get_job("abc")
    assert job.status == "done"
    assert job.completed_at == "2026-06-01T00:05:00+00:00"
    assert job.transcript_path == "transcripts/abc.txt"


def test_update_status_to_error(settings):
    store = Storage(settings.db_path)
    store.create_job("e1", "Bad.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("e1", "error", error="boom")
    job = store.get_job("e1")
    assert job.status == "error"
    assert job.error == "boom"


def test_list_jobs_newest_first(settings):
    store = Storage(settings.db_path)
    store.create_job("a", "A.mp4", "2026-06-01T00:00:00+00:00")
    store.create_job("b", "B.mp4", "2026-06-02T00:00:00+00:00")
    ids = [j.id for j in store.list_jobs()]
    assert ids == ["b", "a"]


def test_reconcile_interrupted(settings):
    store = Storage(settings.db_path)
    store.create_job("q", "Q.mp4", "2026-06-01T00:00:00+00:00")  # queued
    store.create_job("r", "R.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("r", "running")
    store.create_job("d", "D.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("d", "done")
    count = store.reconcile_interrupted()
    assert count == 2
    assert store.get_job("q").status == "interrupted"
    assert store.get_job("r").status == "interrupted"
    assert store.get_job("d").status == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.storage'`.

- [ ] **Step 3: Implement storage**

`web/storage.py`:
```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALID_STATUSES = {"queued", "running", "done", "error", "interrupted"}


@dataclass
class Job:
    id: str
    original_name: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    transcript_path: Optional[str] = None
    error: Optional[str] = None


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        original_name=row["original_name"],
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        transcript_path=row["transcript_path"],
        error=row["error"],
    )


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    transcript_path TEXT,
                    error TEXT
                )
                """
            )

    def create_job(self, job_id: str, original_name: str, created_at: str) -> Job:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, original_name, status, created_at) "
                "VALUES (?, ?, 'queued', ?)",
                (job_id, original_name, created_at),
            )
        return Job(id=job_id, original_name=original_name, status="queued", created_at=created_at)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC").fetchall()
        return [_row_to_job(r) for r in rows]

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        completed_at: Optional[str] = None,
        transcript_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        sets = ["status = ?"]
        params: list[object] = [status]
        if completed_at is not None:
            sets.append("completed_at = ?")
            params.append(completed_at)
        if transcript_path is not None:
            sets.append("transcript_path = ?")
            params.append(transcript_path)
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        params.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)

    def reconcile_interrupted(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'interrupted' "
                "WHERE status IN ('queued', 'running')"
            )
            return cur.rowcount
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_storage.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add web/storage.py tests/test_storage.py
git commit -m "feat: SQLite storage layer for job history"
```

---

## Task 4: Job manager — run lifecycle + log buffers

**Files:**
- Create: `web/jobs.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

`tests/test_jobs.py`:
```python
from pathlib import Path

from web.jobs import JobManager
from web.storage import Storage


def _make_video(settings, job_id="j1", name="clip.mp4"):
    d = settings.uploads_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"x")
    return p


def test_run_success(settings):
    store = Storage(settings.db_path)
    store.create_job("j1", "clip.mp4", "2026-06-01T00:00:00+00:00")
    video = _make_video(settings)

    def fake_transcribe(video_path, cfg, logger=None, out_path=None):
        logger("hello")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("transcript text", encoding="utf-8")
        return Path(out_path)

    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=fake_transcribe)
    mgr._run("j1", video)

    job = store.get_job("j1")
    assert job.status == "done"
    assert job.transcript_path == "transcripts/j1.txt"
    assert (settings.transcripts_dir / "j1.txt").read_text(encoding="utf-8") == "transcript text"
    lines, nxt = mgr.get_logs("j1", since=0)
    assert "hello" in lines
    assert nxt == len(lines)


def test_run_error(settings):
    store = Storage(settings.db_path)
    store.create_job("j2", "bad.mp4", "2026-06-01T00:00:00+00:00")
    video = _make_video(settings, job_id="j2", name="bad.mp4")

    def boom(video_path, cfg, logger=None, out_path=None):
        raise RuntimeError("ffmpeg exploded")

    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=boom)
    mgr._run("j2", video)

    job = store.get_job("j2")
    assert job.status == "error"
    assert "ffmpeg exploded" in job.error


def test_get_logs_since(settings):
    store = Storage(settings.db_path)
    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=lambda *a, **k: None)
    mgr._append_log("x", "a")
    mgr._append_log("x", "b")
    mgr._append_log("x", "c")
    lines, nxt = mgr.get_logs("x", since=1)
    assert lines == ["b", "c"]
    assert nxt == 3
    empty, nxt2 = mgr.get_logs("x", since=3)
    assert empty == []
    assert nxt2 == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.jobs'`.

- [ ] **Step 3: Implement the job manager (run + logs)**

`web/jobs.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_jobs.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add web/jobs.py tests/test_jobs.py
git commit -m "feat: job manager with run lifecycle and log buffers"
```

---

## Task 5: Retention cleanup

**Files:**
- Modify: `web/jobs.py` (add `purge_old_videos` + `start_cleanup_thread`)
- Test: `tests/test_retention.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retention.py`:
```python
import os

from web.jobs import purge_old_videos


def test_purge_removes_only_old_dirs(settings):
    old = settings.uploads_dir / "old"
    old.mkdir(parents=True)
    (old / "v.mp4").write_bytes(b"x")
    new = settings.uploads_dir / "new"
    new.mkdir(parents=True)
    (new / "v.mp4").write_bytes(b"x")

    now = 1_000_000_000
    old_mtime = now - 40 * 86400  # 40 days old, beyond 30-day retention
    os.utime(old, (old_mtime, old_mtime))
    new_mtime = now - 5 * 86400
    os.utime(new, (new_mtime, new_mtime))

    removed = purge_old_videos(settings, now=now)

    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_purge_no_uploads_dir(tmp_path, settings):
    import shutil
    shutil.rmtree(settings.uploads_dir)
    assert purge_old_videos(settings, now=1_000_000_000) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_retention.py -v`
Expected: FAIL — `ImportError: cannot import name 'purge_old_videos'`.

- [ ] **Step 3: Add retention helpers to `web/jobs.py`**

Append these imports near the top of `web/jobs.py` (add to the existing import block):
```python
import shutil
import time
```

Append at the end of `web/jobs.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_retention.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add web/jobs.py tests/test_retention.py
git commit -m "feat: auto-purge uploaded videos past retention window"
```

---

## Task 6: App factory + health + index page

**Files:**
- Modify: `web/__init__.py` (replace placeholder with the factory)
- Create: `web/routes.py`
- Create: `web/templates/base.html`, `web/templates/index.html`
- Create: `web/static/style.css`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

`tests/test_app.py`:
```python
import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    """Records submit calls; never does real work."""

    def __init__(self):
        self.submitted = []
        self._logs = {}

    def submit(self, job_id, video_path):
        self.submitted.append((job_id, str(video_path)))

    def get_logs(self, job_id, since=0):
        lines = self._logs.get(job_id, [])
        return list(lines[since:]), len(lines)


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    client = app.test_client()
    client._storage = storage  # type: ignore[attr-defined]
    client._jobs = jobs  # type: ignore[attr-defined]
    return client


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert b"ok" in resp.data


def test_index_empty(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Upload" in resp.data


def test_index_lists_jobs(client):
    client._storage.create_job("a", "Alpha.mp4", "2026-06-01T00:00:00+00:00")
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Alpha.mp4" in resp.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_app.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_app' from 'web'`.

- [ ] **Step 3: Implement the app factory**

`web/__init__.py` (replace the placeholder entirely):
```python
from __future__ import annotations

from typing import Optional

from flask import Flask

from .jobs import JobManager, start_cleanup_thread
from .routes import register_routes
from .settings import WebSettings
from .storage import Storage


def create_app(
    settings: Optional[WebSettings] = None,
    storage: Optional[Storage] = None,
    job_manager: Optional[object] = None,
    start_background: bool = True,
) -> Flask:
    settings = settings or WebSettings.load()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)

    storage = storage or Storage(settings.db_path)
    storage.reconcile_interrupted()

    if job_manager is None:
        from app.config import Config
        from app.main import transcribe_video

        job_manager = JobManager(
            storage=storage,
            settings=settings,
            cfg_loader=Config.load,
            transcribe_fn=transcribe_video,
        )

    app = Flask(__name__)
    app.secret_key = "transcriber-local"  # only used for flash messages on a private tailnet
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_mb * 1024 * 1024
    app.config["SETTINGS"] = settings
    app.config["STORAGE"] = storage
    app.config["JOBS"] = job_manager

    register_routes(app)

    if start_background:
        start_cleanup_thread(settings)

    return app
```

- [ ] **Step 4: Implement routes (health + index for now)**

`web/routes.py`:
```python
from __future__ import annotations

from flask import Flask, current_app, render_template


def register_routes(app: Flask) -> None:
    @app.route("/healthz")
    def healthz():  # noqa: ANN202
        return "ok", 200

    @app.route("/")
    def index():  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        jobs = storage.list_jobs()[:10]
        return render_template("index.html", jobs=jobs)
```

- [ ] **Step 5: Create base + index templates**

`web/templates/base.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Video Transcriber{% endblock %}</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <header>
    <a href="{{ url_for('index') }}">Video Transcriber</a>
    <span>·</span>
    <a href="{{ url_for('history') }}">History</a>
  </header>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul class="flash">{% for m in messages %}<li>{{ m }}</li>{% endfor %}</ul>
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
  </main>
</body>
</html>
```

`web/templates/index.html`:
```html
{% extends "base.html" %}
{% block content %}
<h1>Upload a video to transcribe</h1>
<form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
  <input type="file" name="video" accept="video/*,audio/*" required>
  <button type="submit">Start transcription</button>
</form>

<h2>Recent transcripts</h2>
<ul class="recent">
{% for job in jobs %}
  <li>
    <a href="{{ url_for('job_page', job_id=job.id) }}">{{ job.original_name }}</a>
    — {{ job.status }}
    {% if job.status == 'done' %}
      (<a href="{{ url_for('download', job_id=job.id) }}">download</a>)
    {% endif %}
  </li>
{% else %}
  <li>None yet.</li>
{% endfor %}
</ul>
{% endblock %}
```

> Note: `index.html` references `url_for('upload')`, `url_for('job_page')`, `url_for('history')`, and `url_for('download')`. Those routes are added in Tasks 7–9. Because Jinja resolves `url_for` at render time, the index page test passes now only if those endpoints exist. To keep this task green, also add the placeholder endpoints in Step 6 below; they are fully implemented in later tasks.

- [ ] **Step 6: Add placeholder endpoints so `url_for` resolves**

Append to `web/routes.py` inside `register_routes` (these are replaced with real logic in Tasks 7–9):
```python
    @app.route("/upload", methods=["POST"])
    def upload():  # noqa: ANN202
        return "", 501

    @app.route("/job/<job_id>")
    def job_page(job_id):  # noqa: ANN202
        return "", 501

    @app.route("/api/job/<job_id>")
    def job_api(job_id):  # noqa: ANN202
        return "", 501

    @app.route("/download/<job_id>")
    def download(job_id):  # noqa: ANN202
        return "", 501

    @app.route("/history")
    def history():  # noqa: ANN202
        return "", 501
```

- [ ] **Step 7: Create minimal CSS**

`web/static/style.css`:
```css
body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
header a { text-decoration: none; margin-right: .25rem; }
header { margin-bottom: 1.5rem; color: #555; }
h1 { font-size: 1.4rem; }
.flash { background: #fde2e1; border: 1px solid #f5b5b1; padding: .5rem 1rem; border-radius: 6px; }
.recent { line-height: 1.8; }
#log { background: #0f1117; color: #d6deeb; padding: 1rem; border-radius: 6px; min-height: 8rem; white-space: pre-wrap; }
button { padding: .4rem .9rem; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: .4rem .5rem; border-bottom: 1px solid #eee; }
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_app.py -v`
Expected: PASS (3 passed).

- [ ] **Step 9: Commit**

```bash
git add web/__init__.py web/routes.py web/templates/base.html web/templates/index.html web/static/style.css tests/test_app.py
git commit -m "feat: Flask app factory, health check, and upload/index page"
```

---

## Task 7: Upload route

**Files:**
- Modify: `web/routes.py` (replace `upload` placeholder)
- Test: `tests/test_upload.py`

- [ ] **Step 1: Write the failing test**

`tests/test_upload.py`:
```python
import io

import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    def __init__(self):
        self.submitted = []

    def submit(self, job_id, video_path):
        self.submitted.append((job_id, str(video_path)))

    def get_logs(self, job_id, since=0):
        return [], 0


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    c = app.test_client()
    c._storage = storage
    c._jobs = jobs
    c._settings = settings
    return c


def test_upload_valid_creates_job_and_redirects(client):
    data = {"video": (io.BytesIO(b"fake video bytes"), "Meeting.mp4")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/job/" in resp.headers["Location"]
    # exactly one job created and submitted
    jobs = client._storage.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].original_name == "Meeting.mp4"
    assert len(client._jobs.submitted) == 1
    # video saved on disk
    job_id = jobs[0].id
    saved = client._settings.uploads_dir / job_id
    assert any(saved.iterdir())


def test_upload_missing_file_flashes_and_redirects(client):
    resp = client.post("/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert client._storage.list_jobs() == []


def test_upload_bad_extension_rejected(client):
    data = {"video": (io.BytesIO(b"x"), "notes.txt")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert client._storage.list_jobs() == []
    assert client._jobs.submitted == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_upload.py -v`
Expected: FAIL — upload returns 501, so assertions on redirect/job creation fail.

- [ ] **Step 3: Replace the `upload` placeholder with the real implementation**

In `web/routes.py`, update the imports at the top:
```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename
```

Replace the placeholder `upload` function with:
```python
    @app.route("/upload", methods=["POST"])
    def upload():  # noqa: ANN202
        settings = current_app.config["SETTINGS"]
        storage = current_app.config["STORAGE"]
        jobs = current_app.config["JOBS"]

        file = request.files.get("video")
        if file is None or not file.filename:
            flash("Please choose a file to upload.")
            return redirect(url_for("index"))

        filename = file.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in settings.allowed_ext:
            flash(f"Unsupported file type: .{ext}")
            return redirect(url_for("index"))

        job_id = uuid.uuid4().hex
        safe_name = secure_filename(filename) or f"upload.{ext}"
        dest_dir = settings.uploads_dir / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        video_path = dest_dir / safe_name
        file.save(str(video_path))

        storage.create_job(job_id, filename, datetime.now(timezone.utc).isoformat())
        jobs.submit(job_id, video_path)
        return redirect(url_for("job_page", job_id=job_id))
```

- [ ] **Step 4: Add the 413 (too large) error handler**

Append inside `register_routes` (after the routes):
```python
    @app.errorhandler(413)
    def too_large(_e):  # noqa: ANN202
        flash("File too large.")
        return redirect(url_for("index"))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_upload.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add web/routes.py tests/test_upload.py
git commit -m "feat: upload route with validation and job submission"
```

---

## Task 8: Job page, status API, and download

**Files:**
- Modify: `web/routes.py` (replace `job_page`, `job_api`, `download` placeholders)
- Create: `web/templates/job.html`, `web/static/job.js`
- Test: `tests/test_job_api.py`

- [ ] **Step 1: Write the failing test**

`tests/test_job_api.py`:
```python
import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    def __init__(self):
        self._logs = {}

    def submit(self, job_id, video_path):
        pass

    def set_logs(self, job_id, lines):
        self._logs[job_id] = lines

    def get_logs(self, job_id, since=0):
        lines = self._logs.get(job_id, [])
        return list(lines[since:]), len(lines)


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    c = app.test_client()
    c._storage = storage
    c._jobs = jobs
    c._settings = settings
    return c


def test_job_page_renders(client):
    client._storage.create_job("j1", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    resp = client.get("/job/j1")
    assert resp.status_code == 200
    assert b"Clip.mp4" in resp.data


def test_job_page_404(client):
    assert client.get("/job/missing").status_code == 404


def test_job_api_streams_status_and_lines(client):
    client._storage.create_job("j1", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j1", "running")
    client._jobs.set_logs("j1", ["one", "two", "three"])
    resp = client.get("/api/job/j1?since=1")
    body = resp.get_json()
    assert body["status"] == "running"
    assert body["lines"] == ["two", "three"]
    assert body["next_index"] == 3
    assert body["download_ready"] is False


def test_job_api_download_ready_when_done(client):
    client._storage.create_job("j2", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j2", "done", transcript_path="transcripts/j2.txt")
    body = client.get("/api/job/j2").get_json()
    assert body["download_ready"] is True


def test_download_serves_transcript(client):
    (client._settings.transcripts_dir / "j3.txt").write_text("the transcript", encoding="utf-8")
    client._storage.create_job("j3", "My Meeting.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j3", "done", transcript_path="transcripts/j3.txt")
    resp = client.get("/download/j3")
    assert resp.status_code == 200
    assert resp.data == b"the transcript"
    assert "attachment" in resp.headers["Content-Disposition"]
    assert "My Meeting.txt" in resp.headers["Content-Disposition"]


def test_download_404_when_no_transcript(client):
    client._storage.create_job("j4", "x.mp4", "2026-06-01T00:00:00+00:00")
    assert client.get("/download/j4").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_job_api.py -v`
Expected: FAIL — placeholders return 501.

- [ ] **Step 3: Add `send_file`, `jsonify`, `Path` to imports**

In `web/routes.py`, extend the flask import to include `jsonify` and `send_file`, and add a `pathlib` import:
```python
from pathlib import Path

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
```

- [ ] **Step 4: Replace the three placeholders with real implementations**

In `web/routes.py`, replace the `job_page`, `job_api`, and `download` placeholder functions with:
```python
    @app.route("/job/<job_id>")
    def job_page(job_id):  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        job = storage.get_job(job_id)
        if job is None:
            abort(404)
        return render_template("job.html", job=job)

    @app.route("/api/job/<job_id>")
    def job_api(job_id):  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        jobs = current_app.config["JOBS"]
        job = storage.get_job(job_id)
        if job is None:
            abort(404)
        since = request.args.get("since", default=0, type=int)
        lines, next_index = jobs.get_logs(job_id, since=since)
        return jsonify(
            {
                "status": job.status,
                "lines": lines,
                "next_index": next_index,
                "download_ready": job.status == "done",
                "error": job.error,
            }
        )

    @app.route("/download/<job_id>")
    def download(job_id):  # noqa: ANN202
        settings = current_app.config["SETTINGS"]
        storage = current_app.config["STORAGE"]
        job = storage.get_job(job_id)
        if job is None or not job.transcript_path:
            abort(404)
        path = settings.data_dir / job.transcript_path
        if not path.exists():
            abort(404)
        download_name = Path(job.original_name).stem + ".txt"
        return send_file(
            str(path),
            as_attachment=True,
            download_name=download_name,
            mimetype="text/plain",
        )
```

- [ ] **Step 5: Create the job page template**

`web/templates/job.html`:
```html
{% extends "base.html" %}
{% block title %}{{ job.original_name }} — Video Transcriber{% endblock %}
{% block content %}
<h1>{{ job.original_name }}</h1>
<p>Status: <span id="status">{{ job.status }}</span></p>
<pre id="log"></pre>
<p id="download" {% if job.status != 'done' %}hidden{% endif %}>
  <a href="{{ url_for('download', job_id=job.id) }}">Download transcript (.txt)</a>
</p>
<script src="{{ url_for('static', filename='job.js') }}"
        data-api="{{ url_for('job_api', job_id=job.id) }}"></script>
{% endblock %}
```

- [ ] **Step 6: Create the progress poller**

`web/static/job.js`:
```javascript
(function () {
  var script = document.currentScript;
  var api = script.getAttribute("data-api");
  var logEl = document.getElementById("log");
  var statusEl = document.getElementById("status");
  var downloadEl = document.getElementById("download");
  var since = 0;
  var terminal = { done: 1, error: 1, interrupted: 1 };

  function poll() {
    fetch(api + "?since=" + since)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.lines && data.lines.length) {
          data.lines.forEach(function (line) { logEl.textContent += line + "\n"; });
          since = data.next_index;
        }
        statusEl.textContent = data.status;
        if (data.download_ready) { downloadEl.removeAttribute("hidden"); }
        if (terminal[data.status]) { return; }
        setTimeout(poll, 1000);
      })
      .catch(function () { setTimeout(poll, 2000); });
  }
  poll();
})();
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_job_api.py -v`
Expected: PASS (6 passed).

- [ ] **Step 8: Commit**

```bash
git add web/routes.py web/templates/job.html web/static/job.js tests/test_job_api.py
git commit -m "feat: job progress page, status API, and transcript download"
```

---

## Task 9: History page

**Files:**
- Modify: `web/routes.py` (replace `history` placeholder)
- Create: `web/templates/history.html`
- Test: `tests/test_history.py`

- [ ] **Step 1: Write the failing test**

`tests/test_history.py`:
```python
import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    def submit(self, job_id, video_path):
        pass

    def get_logs(self, job_id, since=0):
        return [], 0


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    app = create_app(settings=settings, storage=storage, job_manager=FakeJobs(), start_background=False)
    app.config.update(TESTING=True)
    c = app.test_client()
    c._storage = storage
    return c


def test_history_lists_all_jobs(client):
    client._storage.create_job("a", "Alpha.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.create_job("b", "Beta.mp4", "2026-06-02T00:00:00+00:00")
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"Alpha.mp4" in resp.data
    assert b"Beta.mp4" in resp.data


def test_history_empty(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"None yet" in resp.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_history.py -v`
Expected: FAIL — `history` placeholder returns 501.

- [ ] **Step 3: Replace the `history` placeholder**

In `web/routes.py`, replace the placeholder `history` function with:
```python
    @app.route("/history")
    def history():  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        return render_template("history.html", jobs=storage.list_jobs())
```

- [ ] **Step 4: Create the history template**

`web/templates/history.html`:
```html
{% extends "base.html" %}
{% block title %}History — Video Transcriber{% endblock %}
{% block content %}
<h1>History</h1>
<table>
  <thead><tr><th>File</th><th>Status</th><th>Created (UTC)</th><th></th></tr></thead>
  <tbody>
  {% for job in jobs %}
    <tr>
      <td><a href="{{ url_for('job_page', job_id=job.id) }}">{{ job.original_name }}</a></td>
      <td>{{ job.status }}</td>
      <td>{{ job.created_at }}</td>
      <td>
        {% if job.status == 'done' %}
          <a href="{{ url_for('download', job_id=job.id) }}">download</a>
        {% endif %}
      </td>
    </tr>
  {% else %}
    <tr><td colspan="4">None yet.</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endblock %}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_history.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add web/routes.py web/templates/history.html tests/test_history.py
git commit -m "feat: history page listing all transcripts"
```

---

## Task 10: WSGI entrypoint + web requirements

**Files:**
- Create: `web/wsgi.py`
- Create: `requirements-web.txt`
- Test: `tests/test_wsgi_import.py`

- [ ] **Step 1: Create the web requirements file**

`requirements-web.txt`:
```
flask>=3.0
gunicorn>=21.2
```

- [ ] **Step 2: Write the failing test**

`tests/test_wsgi_import.py`:
```python
def test_wsgi_exposes_app(settings, monkeypatch):
    # Point DATA_DIR at a temp dir so create_app() doesn't touch /data.
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import importlib
    import web.wsgi as wsgi
    importlib.reload(wsgi)
    assert wsgi.app is not None
    client = wsgi.app.test_client()
    assert client.get("/healthz").status_code == 200
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_wsgi_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.wsgi'`.

- [ ] **Step 4: Implement the WSGI entrypoint**

`web/wsgi.py`:
```python
from web import create_app

app = create_app()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_wsgi_import.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add web/wsgi.py requirements-web.txt tests/test_wsgi_import.py
git commit -m "feat: gunicorn WSGI entrypoint and web requirements"
```

---

## Task 11: Docker image + compose

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`

(No automated test — verification is `docker compose config` if Docker is available locally, otherwise the build is verified on the VPS during deploy.)

- [ ] **Step 1: Create the Dockerfile**

`Dockerfile`:
```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY app/ ./app/
COPY web/ ./web/

ENV DATA_DIR=/data
EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "1200", \
     "--bind", "0.0.0.0:8000", "web.wsgi:app"]
```

- [ ] **Step 2: Create the .dockerignore**

`.dockerignore`:
```
.venv/
.git/
.gitignore
data/
docs/
tests/
**/__pycache__/
*.pyc
.env
start_app.bat
*.mp4
*.mkv
*.mov
*.avi
*.webm
```

- [ ] **Step 3: Create docker-compose.yml**

`docker-compose.yml`:
```yaml
services:
  web:
    build: .
    image: video-transcriber-web
    container_name: transcriber
    restart: unless-stopped
    env_file:
      - .env
    environment:
      - DATA_DIR=/data
      - RETAIN_VIDEO_DAYS=30
      - MAX_CONTENT_MB=2048
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./data:/data
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else sys.exit(1)"]
      interval: 30s
      timeout: 5s
      retries: 3
```

- [ ] **Step 4: Validate compose syntax (if Docker is available)**

Run: `docker compose config`
Expected: prints the resolved config with no error. (If Docker is not installed locally, skip — this is validated on the VPS in Task 13.)

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml
git commit -m "feat: Docker image and compose for the web service"
```

---

## Task 12: Deploy script + documentation

**Files:**
- Create: `deploy.sh`
- Create: `docs/deploy.md`
- Modify: `README.md`

- [ ] **Step 1: Create the deploy script**

`deploy.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found in $(pwd)." >&2
  echo "Create it with your key, e.g.:  echo 'GROQ_API_KEY=gsk_...' > .env" >&2
  exit 1
fi

echo "==> Pulling latest code..."
git pull --ff-only

echo "==> Building and starting the container..."
docker compose up -d --build

echo "==> Container status:"
docker compose ps

cat <<'EOF'

==> Done.

First-time only — expose the app on your tailnet (run once on the host):
  sudo tailscale serve --bg --https=443 http://127.0.0.1:8000

Optional — give it the short domain by renaming this machine to "transcribe":
  sudo tailscale set --hostname=transcribe

Then open:  https://transcribe.<your-tailnet>.ts.net

To view logs:        docker compose logs -f
To redeploy later:   ./deploy.sh
EOF
```

- [ ] **Step 2: Mark it executable and commit-friendly**

Run: `git update-index --chmod=+x deploy.sh` (after `git add` in Step 5), or on the VPS run `chmod +x deploy.sh`. Note this in `docs/deploy.md`.

- [ ] **Step 3: Create the deploy documentation**

`docs/deploy.md`:
```markdown
# Deploying the Web Transcriber to a VPS

The app runs as a Docker container, reachable only over your Tailscale tailnet.

## Prerequisites (on the VPS)
- Docker + the `docker compose` plugin
- Tailscale installed and logged in (`tailscale status` shows the machine)

## One-time setup
1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/tanurus/video-transcriber.git
   cd video-transcriber
   ```
2. Create `.env` with your Groq key (never committed):
   ```bash
   echo 'GROQ_API_KEY=gsk_your_key_here' > .env
   ```
3. Make the deploy script executable:
   ```bash
   chmod +x deploy.sh
   ```
4. Deploy:
   ```bash
   ./deploy.sh
   ```
5. Expose it on the tailnet (run once):
   ```bash
   sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
   ```
6. (Optional) Rename the machine for a short domain:
   ```bash
   sudo tailscale set --hostname=transcribe
   ```
7. Open `https://transcribe.<your-tailnet>.ts.net` from any device on your tailnet.

## Redeploying after changes
```bash
./deploy.sh
```

## Where data lives
- `./data/app.db` — job history (SQLite)
- `./data/uploads/<job-id>/` — uploaded videos (auto-deleted after `RETAIN_VIDEO_DAYS`, default 30)
- `./data/transcripts/<job-id>.txt` — transcripts (kept forever)

## Configuration (set in `docker-compose.yml` or `.env`)
- `RETAIN_VIDEO_DAYS` (default 30)
- `MAX_CONTENT_MB` (default 2048)
- `ALLOWED_EXT` (default `mp4,mkv,mov,avi,webm,m4a,mp3,wav`)

## Notes
- The container binds only to `127.0.0.1:8000`; nothing is exposed to the public internet.
- Do **not** enable `tailscale funnel` unless you also add authentication — the app has none by design (the private tailnet is the access boundary).
```

- [ ] **Step 4: Add a web section to README.md**

Add this section to `README.md` (after the existing "## Setup" content, before "## Notes"):
```markdown
## Web app on a VPS (browser access over Tailscale)

You can run this as a browser app on a Linux VPS and reach it from any device on
your Tailscale tailnet — upload a video, watch live progress, and download the
transcript, with no local app running.

See [docs/deploy.md](docs/deploy.md) for full instructions. In short:

```bash
git clone https://github.com/tanurus/video-transcriber.git && cd video-transcriber
echo 'GROQ_API_KEY=gsk_...' > .env
chmod +x deploy.sh && ./deploy.sh
sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
```

Then open `https://transcribe.<your-tailnet>.ts.net`.
```

- [ ] **Step 5: Commit**

```bash
git add deploy.sh docs/deploy.md README.md
git update-index --chmod=+x deploy.sh
git commit -m "docs: deploy script and VPS deployment instructions"
```

---

## Task 13: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: ALL tests pass.

- [ ] **Step 2: Run the app locally end-to-end (smoke, no Docker)**

Run:
```powershell
$env:DATA_DIR = "$PWD\localdata"
$env:GROQ_API_KEY = "<your real groq key>"
.\.venv\Scripts\python.exe -m flask --app web.wsgi:app run --port 8000
```
Open `http://127.0.0.1:8000`, upload a short clip, confirm: live log streams, status reaches `done`, the Download button appears and returns the `.txt`, and the file appears under `localdata/transcripts/`. (ffmpeg must be on PATH locally for this smoke test — the desktop launcher's PATH applies.)
Stop with Ctrl+C. This local data dir is git-ignored (see next step).

- [ ] **Step 3: Ensure local/runtime artifacts are git-ignored**

Confirm `.gitignore` excludes the runtime data directories. Add these lines if missing:
```
data/
localdata/
```
Then:
```bash
git add .gitignore
git commit -m "chore: ignore local runtime data directories"
```

- [ ] **Step 4: Push to GitHub**

```bash
git push
```

- [ ] **Step 5: Deploy on the VPS** (performed by the user, per the spec)

Follow `docs/deploy.md`: clone/pull, create `.env`, `./deploy.sh`, `tailscale serve`, browse to the tailnet domain. Verify a real upload transcribes end-to-end.

---

## Self-Review Notes

- **Spec coverage:** topology (Tasks 11–12), `127.0.0.1`-only publish + Tailscale serve (Tasks 11–12), storage layout & SQLite (Tasks 3, 11), `out_path` core reuse (Task 1), single-slot worker + live log polling (Tasks 4, 8), retention purge (Task 5), interrupted-job reconciliation (Tasks 3, 6), upload validation + size limit (Task 7), history (Task 9), `.env` never imaged (`.dockerignore`, Task 11), deploy-by-script (Task 12). All spec sections map to a task.
- **No placeholders:** every code step contains complete code; the temporary route stubs in Task 6 are explicitly replaced in Tasks 7–9 and called out as such.
- **Type/name consistency:** endpoint names (`index`, `upload`, `job_page`, `job_api`, `download`, `history`, `healthz`) are used identically across routes, templates, and tests; `JobManager.get_logs` returns `(lines, next_index)` everywhere; `Storage.update_status` keyword args match all call sites; `WebSettings` properties (`uploads_dir`, `transcripts_dir`, `db_path`) are consistent across tasks.
```
