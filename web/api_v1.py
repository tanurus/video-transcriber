"""Token-authenticated API for other sources: phone shortcuts, n8n, scripts, the desktop app.

    curl -H "Authorization: Bearer vt_..." -F file=@meeting.m4a -F preset=accuracy \\
         http://<host>:18920/api/v1/jobs
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from .intake import device_label, ingest, recorded_at_from_epoch_ms
from .routes import job_json

bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


@bp.before_request
def _auth():  # noqa: ANN202
    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else request.args.get("token", "")
    if not token:
        return jsonify({"error": "Missing API token (Settings → API tokens)."}), 401
    record = current_app.config["STORAGE"].token_by_hash(hashlib.sha256(token.encode()).hexdigest())
    if record is None:
        return jsonify({"error": "Invalid or revoked API token."}), 401
    request.environ["vt.token_name"] = record.name
    return None


@bp.post("/jobs")
def create_job():  # noqa: ANN202
    settings = current_app.config["SETTINGS"]
    storage = current_app.config["STORAGE"]
    jobs = current_app.config["JOBS"]
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Send the recording as multipart field 'file'."}), 400
    ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
    if ext not in settings.allowed_ext:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400
    try:
        raw = json.loads(request.form.get("options") or "{}")
        if not isinstance(raw, dict):
            raise ValueError
    except ValueError:
        return jsonify({"error": "'options' must be a JSON object."}), 400
    for key in ("profile_id", "language", "prompt", "hotwords", "model"):
        if request.form.get(key):
            raw.setdefault(key, request.form[key])
    fd, tmp = tempfile.mkstemp(dir=settings.data_dir)
    with open(fd, "wb") as out:
        upload.save(out)
    recorded = request.form.get("recorded_at") or recorded_at_from_epoch_ms(request.form.get("last_modified_ms"))
    detail = request.form.get("source_detail") or request.environ.get("vt.token_name", "")
    ua = device_label(request.headers.get("User-Agent", ""))
    result = ingest(
        storage, settings.uploads_dir, Path(tmp), request.form.get("filename") or upload.filename,
        raw_options=raw, preset=request.form.get("preset"),
        source=(request.form.get("source") or "api")[:40],
        source_detail=f"{detail} ({ua})" if ua and ua not in detail else detail,
        recorded_at=recorded, title=request.form.get("title"),
        on_duplicate=request.form.get("on_duplicate", "skip"),
    )
    if not result.duplicate:
        jobs.submit(result.job_id)
    job = storage.get_job(result.job_id)
    return jsonify({"duplicate": result.duplicate, "job": job_json(job)}), (200 if result.duplicate else 201)


@bp.get("/jobs/<job_id>")
def get_job(job_id):  # noqa: ANN202
    storage = current_app.config["STORAGE"]
    jobs = current_app.config["JOBS"]
    job = storage.get_job(job_id)
    if job is None:
        return jsonify({"error": "not found"}), 404
    data = job_json(job)
    if job.status == "done" and request.args.get("text") == "1":
        data["transcript_raw"] = jobs.read_text(job.transcript_path)
        data["transcript_clean"] = jobs.read_text(job.clean_text_path)
    return jsonify(data)


@bp.get("/jobs")
def list_jobs():  # noqa: ANN202
    storage = current_app.config["STORAGE"]
    limit = min(200, max(1, request.args.get("limit", default=50, type=int)))
    return jsonify({"jobs": [job_json(j) for j in storage.search_jobs(limit=limit)]})
