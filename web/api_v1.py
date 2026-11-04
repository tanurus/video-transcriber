"""Token-authenticated API for other sources: phone shortcuts, n8n, scripts, the desktop app.

    curl -H "Authorization: Bearer vt_..." -F file=@meeting.m4a -F preset=accuracy \\
         http://<host>:18920/api/v1/jobs

``-F pipeline=validated`` stores the recording without transcribing it, for the
chunk-level pipeline (/api/v2) to pick up.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from .auth import require_token
from .intake import ChunkedUploads, UploadError, device_label, ingest, recorded_at_from_epoch_ms, recorded_at_from_name
from .routes import job_json

bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


@bp.before_request
def _auth():  # noqa: ANN202
    return require_token()


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
    pipeline = (request.form.get("pipeline") or "standard").lower()
    if not result.duplicate:
        if pipeline == "validated":
            # Held for the chunk-level pipeline (n8n opens a /api/v2 session on it).
            storage.update_status(result.job_id, "received")
        else:
            jobs.submit(result.job_id)
    job = storage.get_job(result.job_id)
    return jsonify({"duplicate": result.duplicate, "pipeline": pipeline, "job": job_json(job)}), (200 if result.duplicate else 201)


@bp.get("/jobs/<job_id>")
def get_job(job_id):  # noqa: ANN202
    storage = current_app.config["STORAGE"]
    jobs = current_app.config["JOBS"]
    job = storage.get_job(job_id)
    if job is None:
        return jsonify({"error": "not found"}), 404
    data = job_json(job)
    data["settings"] = {k: v for k, v in job.settings.items() if not k.startswith("_")}
    if job.status == "done" and request.args.get("text") == "1":
        data["transcript_raw"] = jobs.read_text(job.transcript_path)
        data["transcript_clean"] = jobs.read_text(job.clean_text_path)
    return jsonify(data)


@bp.get("/jobs")
def list_jobs():  # noqa: ANN202
    storage = current_app.config["STORAGE"]
    limit = min(200, max(1, request.args.get("limit", default=50, type=int)))
    found = storage.search_jobs(
        text=(request.args.get("q") or "").strip()[:200],
        status=request.args.get("status") or "",
        source=request.args.get("source") or "",
        latest_only=request.args.get("latest_only") in ("1", "true", "yes"),
        limit=limit,
    )
    return jsonify({"jobs": [job_json(j) for j in found]})


# --- resumable uploads (the MCP server uses these for multi-GB recordings) --------------------

def _uploads() -> ChunkedUploads:
    settings = current_app.config["SETTINGS"]
    return ChunkedUploads(settings.data_dir / "incoming", settings.allowed_ext, settings.max_file_mb * 1024 * 1024)


@bp.post("/uploads")
def upload_start():  # noqa: ANN202
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(_uploads().start(str(body.get("filename", "")), int(body.get("size") or 0),
                                        body.get("last_modified"))), 201
    except UploadError as e:
        return jsonify({"error": str(e)}), e.status
    except (TypeError, ValueError):
        return jsonify({"error": "'size' must be a number."}), 400


@bp.get("/uploads/<upload_id>")
def upload_status(upload_id):  # noqa: ANN202
    try:
        return jsonify(_uploads().status(upload_id))
    except UploadError as e:
        return jsonify({"error": str(e)}), e.status


@bp.put("/uploads/<upload_id>")
def upload_chunk(upload_id):  # noqa: ANN202
    try:
        offset = int(request.args.get("offset", "-1"))
        return jsonify(_uploads().append(upload_id, offset, request.get_data(cache=False)))
    except UploadError as e:
        return jsonify({"error": str(e)}), e.status
    except ValueError:
        return jsonify({"error": "bad offset"}), 400


@bp.post("/uploads/<upload_id>/complete")
def upload_complete(upload_id):  # noqa: ANN202
    """Turn a finished upload into a job. Same fields as POST /jobs, as JSON."""
    settings = current_app.config["SETTINGS"]
    storage = current_app.config["STORAGE"]
    jobs = current_app.config["JOBS"]
    body = request.get_json(silent=True) or {}
    raw = body.get("options") or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "'options' must be a JSON object."}), 400
    for key in ("profile_id", "language", "prompt", "hotwords", "model"):
        if body.get(key):
            raw.setdefault(key, body[key])
    up = _uploads()
    try:
        part, meta = up.finish(upload_id)
    except UploadError as e:
        return jsonify({"error": str(e)}), e.status
    name = str(body.get("filename") or meta["filename"])
    recorded = (body.get("recorded_at") or recorded_at_from_name(name)
                or recorded_at_from_epoch_ms(meta.get("last_modified")))
    result = ingest(
        storage, settings.uploads_dir, part, name,
        raw_options=raw, preset=body.get("preset"),
        source=str(body.get("source") or "api")[:40],
        source_detail=str(body.get("source_detail") or request.environ.get("vt.token_name", ""))[:200],
        recorded_at=recorded, title=body.get("title"),
        on_duplicate=body.get("on_duplicate", "skip"),
    )
    up.discard(upload_id)
    if not result.duplicate:
        jobs.submit(result.job_id)
    job = storage.get_job(result.job_id)
    return jsonify({"duplicate": result.duplicate, "job": job_json(job)}), (200 if result.duplicate else 201)
