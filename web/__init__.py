from __future__ import annotations

import os
from typing import Optional

from flask import Flask

from .connections import ConnectionStore
from .jobs import JobManager, start_cleanup_thread
from .routes import register_routes
from .settings import WebSettings
from .storage import Storage


def _requeue_pending(storage: Storage, settings: WebSettings, job_manager) -> None:
    """Resubmit work that was pending when the process died.

    Queued uploads are still on disk, so re-running them beats forcing the user
    to upload multi-GB files again. Jobs whose video is gone are interrupted.
    """
    if hasattr(job_manager, "resume"):
        job_manager.resume()
        return
    # Minimal job managers (tests) only know submit().
    for job in storage.list_jobs():
        if job.status != "queued":
            continue
        job_dir = settings.uploads_dir / job.id
        # Dotfiles are metadata, never the upload.
        files = (
            sorted(p for p in job_dir.iterdir() if p.is_file() and not p.name.startswith("."))
            if job_dir.is_dir()
            else []
        )
        if files:
            job_manager.submit(job.id, files[0])
        else:
            storage.update_status(job.id, "interrupted")


def _default_ai_factory(conns):
    from .ai import AIConfig, AIFinisher

    return AIFinisher(AIConfig(conns.openai_base_url, conns.openai_api_key, conns.openai_model,
                               language=conns.ai_language))


def create_app(
    settings: Optional[WebSettings] = None,
    storage: Optional[Storage] = None,
    job_manager: Optional[object] = None,
    start_background: bool = True,
    connections: Optional[ConnectionStore] = None,
    ai_factory=None,
) -> Flask:
    settings = settings or WebSettings.load()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)

    storage = storage or Storage(settings.db_path)
    storage.reconcile_interrupted()
    connections = connections or ConnectionStore(settings.data_dir / "connections.json")
    ai_factory = ai_factory or _default_ai_factory

    if job_manager is None:
        from app.config import Config
        from app.pipeline import transcribe_media
        from .supabase_sync import upsert

        job_manager = JobManager(
            storage=storage,
            settings=settings,
            cfg_loader=Config.load,
            transcribe_fn=transcribe_media,
            connections=connections,
            ai_factory=ai_factory,
            sync_fn=upsert,
            rnnoise_model=os.getenv("RNNOISE_MODEL") or "/srv/models/sh.rnnn",
        )

    _requeue_pending(storage, settings, job_manager)

    app = Flask(__name__)
    # Flash messages and one-time token display only; on a private tailnet there's
    # no session-security requirement. Prefer an env-provided key.
    app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
    # Per-request cap: chunked uploads send 8 MB pieces; plain form posts can be bigger.
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_mb * 1024 * 1024
    app.config["SETTINGS"] = settings
    app.config["STORAGE"] = storage
    app.config["JOBS"] = job_manager
    app.config["CONNECTIONS"] = connections
    app.config["AI_FACTORY"] = ai_factory

    register_routes(app)
    from .api_v1 import bp as api_v1

    app.register_blueprint(api_v1)

    if start_background:
        start_cleanup_thread(settings, storage)
        if hasattr(job_manager, "start_sync_loop"):
            job_manager.start_sync_loop()

    return app
