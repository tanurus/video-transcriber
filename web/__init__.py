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
