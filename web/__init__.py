from __future__ import annotations

import os
from typing import Optional

from flask import Flask

from .jobs import JobManager, start_cleanup_thread
from .routes import register_routes
from .settings import WebSettings
from .storage import Storage


def _requeue_pending(storage: Storage, settings: WebSettings, job_manager) -> None:
    """Resubmit jobs that were queued (never started) when the process died.

    Their uploads are still on disk, so re-running them beats forcing the user
    to upload multi-GB files again. Jobs whose video is gone are interrupted.
    """
    for job in storage.list_jobs():
        if job.status != "queued":
            continue
        job_dir = settings.uploads_dir / job.id
        # Dotfiles (the per-job .options.json) are metadata, never the upload.
        files = (
            sorted(p for p in job_dir.iterdir() if p.is_file() and not p.name.startswith("."))
            if job_dir.is_dir()
            else []
        )
        if files:
            job_manager.submit(job.id, files[0])
        else:
            storage.update_status(job.id, "interrupted")


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

    _requeue_pending(storage, settings, job_manager)

    app = Flask(__name__)
    # Flash messages only; on a private tailnet there's no session-security requirement.
    # Prefer an env-provided key; fall back to a random per-process key.
    app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_mb * 1024 * 1024
    app.config["SETTINGS"] = settings
    app.config["STORAGE"] = storage
    app.config["JOBS"] = job_manager

    register_routes(app)

    if start_background:
        start_cleanup_thread(settings, storage)

    return app
