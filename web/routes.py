from __future__ import annotations

import uuid
from datetime import datetime, timezone
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
        if since < 0:
            since = 0
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
        path = (settings.data_dir / job.transcript_path).resolve()
        if not path.is_relative_to(settings.data_dir.resolve()):
            abort(404)
        if not path.exists():
            abort(404)
        download_name = Path(job.original_name).stem + ".txt"
        return send_file(
            str(path),
            as_attachment=True,
            download_name=download_name,
            mimetype="text/plain",
        )

    @app.route("/history")
    def history():  # noqa: ANN202
        storage = current_app.config["STORAGE"]
        return render_template("history.html", jobs=storage.list_jobs())
