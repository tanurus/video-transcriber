from __future__ import annotations

import uuid
from datetime import datetime, timezone

from flask import (
    Flask,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename


def register_routes(app: Flask) -> None:
    @app.route("/healthz")
    def healthz():  # noqa: ANN202
        return "ok", 200

    @app.route("/")
    def index():  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        jobs = storage.list_jobs()[:10]
        return render_template("index.html", jobs=jobs)

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

    @app.errorhandler(413)
    def too_large(_e):  # noqa: ANN202
        flash("File too large.")
        return redirect(url_for("index"))

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
