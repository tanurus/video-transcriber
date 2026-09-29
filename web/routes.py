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

from .jobs import QUALITIES, write_options


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

        # The browser uploads via XHR (see static/upload.js) so it can show a
        # progress bar; those requests want JSON, not a redirect. A plain form
        # POST (JS disabled) still gets the original redirect behaviour.
        wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        def fail(message: str, status: int = 400):  # noqa: ANN202
            if wants_json:
                return jsonify({"error": message}), status
            flash(message)
            return redirect(url_for("index"))

        file = request.files.get("video")
        if file is None or not file.filename:
            return fail("Please choose a file to upload.")

        filename = file.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in settings.allowed_ext:
            return fail(f"Unsupported file type: .{ext}")

        job_id = uuid.uuid4().hex
        safe_name = secure_filename(filename)
        # secure_filename strips non-ASCII: "Видео.mp4" becomes "mp4" (no dot).
        # The transcription API sniffs format by extension, so make sure one survives.
        if not safe_name or "." not in safe_name:
            safe_name = f"upload.{ext}"
        dest_dir = settings.uploads_dir / job_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        video_path = dest_dir / safe_name
        file.save(str(video_path))
        quality = request.form.get("quality") or "best"
        if quality in QUALITIES and quality != "best":
            write_options(dest_dir, {"quality": quality})

        storage.create_job(job_id, filename, datetime.now(timezone.utc).isoformat())
        jobs.submit(job_id, video_path)
        job_url = url_for("job_page", job_id=job_id)
        if wants_json:
            return jsonify({"job_id": job_id, "job_url": job_url})
        return redirect(job_url)

    @app.errorhandler(413)
    def too_large(_e):  # noqa: ANN202
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "File too large."}), 413
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
