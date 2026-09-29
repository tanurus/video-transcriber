from __future__ import annotations

import io
import json
import re
import secrets
import hashlib
import tempfile
import zipfile
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from app import options as O

from .intake import (
    ChunkedUploads,
    UploadError,
    device_label,
    ingest,
    recorded_at_from_epoch_ms,
    recorded_at_from_name,
)


def _svc():
    c = current_app.config
    return c["SETTINGS"], c["STORAGE"], c["JOBS"]


def _slug(text: str, limit: int = 80) -> str:
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", text or "").strip()
    return re.sub(r"\s+", " ", text)[:limit].strip() or "transcript"


def download_name(job, suffix: str) -> str:
    """'2026-09-29 — Kayak partnership review.txt' — date + title when known."""
    date = (job.recorded_at or job.created_at or "")[:10]
    base = _slug(job.title) if job.title else _slug(Path(job.original_name).stem)
    return f"{date} — {base}{suffix}" if date else f"{base}{suffix}"


def job_json(job, jobs=None) -> dict:
    return {
        "id": job.id,
        "name": job.display_name,
        "original_name": job.original_name,
        "title": job.title,
        "description": job.description,
        "tags": job.tag_list,
        "status": job.status,
        "error": job.error,
        "ai_status": job.ai_status or "none",
        "ai_error": job.ai_error,
        "sync_status": job.sync_status or "none",
        "sync_error": job.sync_error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "recorded_at": job.recorded_at,
        "duration_sec": job.duration_sec,
        "languages": [x for x in (job.languages or "").split(",") if x],
        "model": job.model,
        "version": job.version or 1,
        "parent_id": job.parent_id,
        "source": job.source,
        "source_detail": job.source_detail,
        "settings_summary": job.settings.get("_summary"),
        "url": url_for("job_page", job_id=job.id),
    }


def _options_context(storage, values=None):
    return {
        "catalog": O.CATALOG,
        "groups": O.GROUPS,
        "presets": O.PRESET_LABELS,
        "values": values if values is not None else O.resolve(base=storage.get_setting("default_options", {}) or {}),
        "profiles": storage.list_profiles(),
        "options_meta": json.dumps({"presets": O.PRESETS, "defaults": O.defaults()}),
    }


def register_routes(app: Flask) -> None:
    @app.context_processor
    def _globals():  # noqa: ANN202
        return {"download_name": download_name}

    @app.route("/healthz")
    def healthz():  # noqa: ANN202
        return "ok", 200

    # --- pages --------------------------------------------------------------------------

    @app.route("/")
    def index():  # noqa: ANN202
        settings, storage, _jobs = _svc()
        conns = current_app.config["CONNECTIONS"].load()
        ctx = _options_context(storage)
        recent = storage.search_jobs(limit=8)
        return render_template("index.html", jobs=recent, conns=conns,
                               allowed=sorted(settings.allowed_ext), **ctx)

    @app.route("/library")
    def library():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        q = request.args.get("q", "").strip()
        status = request.args.get("status", "")
        source = request.args.get("source", "")
        sync = request.args.get("sync", "")
        latest = request.args.get("all_versions") != "1"
        found = storage.search_jobs(text=q, status=status, source=source, sync_status=sync, latest_only=latest)
        conns = current_app.config["CONNECTIONS"].load()
        return render_template("library.html", jobs=found, q=q, status=status, source=source, sync=sync,
                               latest=latest, conns=conns, **_options_context(storage))

    # Old links from v1 keep working.
    @app.route("/history")
    def history():  # noqa: ANN202
        return redirect(url_for("library"))

    @app.route("/job/<job_id>")
    def job_page(job_id):  # noqa: ANN202
        _settings, storage, jobs = _svc()
        job = storage.get_job(job_id)
        if job is None:
            abort(404)
        raw = jobs.read_text(job.transcript_path) or ""
        clean = jobs.read_text(job.clean_text_path)
        segments = jobs.read_segments(job) or []
        from app.pipeline import to_timestamped_text
        conns = current_app.config["CONNECTIONS"].load()
        return render_template(
            "job.html", job=job, raw=raw, clean=clean, timestamped=to_timestamped_text(segments),
            versions=storage.versions_of(job_id), ai_meta=jobs.read_ai_meta(job_id),
            has_source=bool(job.audio_path and Path(job.audio_path).exists()) or bool(jobs.upload_file(storage.root_of(job_id))),
            conns=conns, **_options_context(storage, values=O.resolve(job.settings)),
        )

    # --- uploads (browser) ----------------------------------------------------------------

    def _uploads() -> ChunkedUploads:
        settings = current_app.config["SETTINGS"]
        return ChunkedUploads(settings.data_dir / "incoming", settings.allowed_ext,
                              settings.max_file_mb * 1024 * 1024)

    @app.post("/api/uploads")
    def upload_start():  # noqa: ANN202
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(_uploads().start(str(body.get("filename", "")), int(body.get("size") or 0),
                                            body.get("last_modified")))
        except UploadError as e:
            return jsonify({"error": str(e)}), e.status

    @app.get("/api/uploads/<upload_id>")
    def upload_status(upload_id):  # noqa: ANN202
        try:
            return jsonify(_uploads().status(upload_id))
        except UploadError as e:
            return jsonify({"error": str(e)}), e.status

    @app.put("/api/uploads/<upload_id>")
    def upload_chunk(upload_id):  # noqa: ANN202
        try:
            offset = int(request.args.get("offset", "-1"))
            return jsonify(_uploads().append(upload_id, offset, request.get_data(cache=False)))
        except UploadError as e:
            return jsonify({"error": str(e)}), e.status
        except ValueError:
            return jsonify({"error": "bad offset"}), 400

    @app.post("/api/uploads/<upload_id>/complete")
    def upload_complete(upload_id):  # noqa: ANN202
        settings, storage, jobs = _svc()
        body = request.get_json(silent=True) or {}
        up = _uploads()
        try:
            part, meta = up.finish(upload_id)
        except UploadError as e:
            return jsonify({"error": str(e)}), e.status
        recorded = recorded_at_from_name(meta["filename"]) or recorded_at_from_epoch_ms(meta.get("last_modified"))
        result = ingest(
            storage, settings.uploads_dir, part, meta["filename"],
            raw_options=body.get("options") or {}, preset=body.get("preset"),
            source="web", source_detail=device_label(request.headers.get("User-Agent", "")),
            recorded_at=recorded, on_duplicate=body.get("on_duplicate", "skip"),
        )
        up.discard(upload_id)
        if not result.duplicate:
            jobs.submit(result.job_id)
        job = storage.get_job(result.job_id)
        return jsonify({"duplicate": result.duplicate, "job": job_json(job)})

    # Plain multipart form (no JavaScript) still works.
    @app.post("/upload")
    def upload():  # noqa: ANN202
        settings, storage, jobs = _svc()
        files = [f for f in request.files.getlist("video") if f and f.filename]
        if not files:
            flash("Please choose at least one file.")
            return redirect(url_for("index"))
        raw = O.from_form(request.form)
        preset = request.form.get("preset") or None
        created = []
        for f in files:
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
            if ext not in settings.allowed_ext:
                flash(f"Skipped {f.filename}: unsupported type .{ext}")
                continue
            fd, tmp = tempfile.mkstemp(dir=settings.data_dir)
            with open(fd, "wb") as out:
                f.save(out)
            res = ingest(storage, settings.uploads_dir, Path(tmp), f.filename, raw_options=raw, preset=preset,
                         source="web", source_detail=device_label(request.headers.get("User-Agent", "")))
            if not res.duplicate:
                jobs.submit(res.job_id)
            else:
                flash(f"{f.filename} is already in the library.")
            created.append(res.job_id)
        if len(created) == 1:
            return redirect(url_for("job_page", job_id=created[0]))
        return redirect(url_for("library"))

    @app.errorhandler(413)
    def too_large(_e):  # noqa: ANN202
        if request.path.startswith("/api/"):
            return jsonify({"error": "Too large."}), 413
        flash("File too large.")
        return redirect(url_for("index"))

    # --- job data --------------------------------------------------------------------------

    @app.route("/api/job/<job_id>")
    def job_api(job_id):  # noqa: ANN202
        _settings, storage, jobs = _svc()
        job = storage.get_job(job_id)
        if job is None:
            abort(404)
        since = max(0, request.args.get("since", default=0, type=int))
        lines, next_index = jobs.get_logs(job_id, since=since)
        data = job_json(job)
        data.update({"lines": lines, "next_index": next_index, "download_ready": job.status == "done"})
        return jsonify(data)

    @app.get("/api/jobs")
    def jobs_api():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        ids = [i for i in request.args.get("ids", "").split(",") if i]
        found = storage.get_jobs(ids) if ids else storage.search_jobs(limit=50)
        return jsonify({"jobs": [job_json(j) for j in found], "counts": storage.counts()})

    @app.post("/api/job/<job_id>/meta")
    def job_meta(job_id):  # noqa: ANN202
        _settings, storage, jobs = _svc()
        job = storage.get_job(job_id)
        if job is None:
            abort(404)
        body = request.get_json(silent=True) or {}
        values = {}
        if "title" in body:
            values["title"] = str(body["title"]).strip()[:120] or None
        if "description" in body:
            values["description"] = str(body["description"]).strip()[:2000] or None
        if "tags" in body:
            tags = body["tags"] if isinstance(body["tags"], list) else str(body["tags"]).split(",")
            values["tags"] = json.dumps([t.strip().lower() for t in tags if str(t).strip()][:12], ensure_ascii=False)
        if values:
            storage.update_job(job_id, **values)
            if job.sync_status in ("synced", "error", "pending"):
                jobs.mark_sync([job_id])
        return jsonify(job_json(storage.get_job(job_id)))

    @app.post("/api/jobs/bulk")
    def jobs_bulk():  # noqa: ANN202
        _settings, storage, jobs = _svc()
        body = request.get_json(silent=True) or {}
        action = body.get("action")
        ids = [str(i) for i in (body.get("ids") or [])][:500]
        if not ids:
            return jsonify({"error": "Select at least one transcript."}), 400
        conns = current_app.config["CONNECTIONS"].load()
        if action == "sync":
            if not conns.supabase_ready:
                return jsonify({"error": "Connect Supabase first (Settings → Connections)."}), 400
            n = jobs.mark_sync(ids)
            return jsonify({"message": f"Queued {n} transcript(s) for Supabase.", "count": n})
        if action == "ai":
            if not conns.ai_ready:
                return jsonify({"error": "Add an OpenAI key first (Settings → Connections)."}), 400
            n = jobs.enqueue_ai(ids)
            return jsonify({"message": f"AI finishing queued for {n} transcript(s).", "count": n})
        if action == "regenerate":
            raw = body.get("options") or {}
            preset = body.get("preset") or None
            keep = bool(body.get("keep_base", False))
            created, failed = [], []
            for job_id in ids:
                try:
                    created.append(jobs.regenerate(job_id, raw, preset=preset, keep_base=keep))
                except Exception as e:  # noqa: BLE001
                    failed.append(f"{job_id[:8]}: {e}")
            msg = f"Regenerating {len(created)} transcript(s)."
            if failed:
                msg += " Could not regenerate: " + "; ".join(failed)
            return jsonify({"message": msg, "created": created, "failed": failed})
        if action == "delete":
            n = jobs.delete(ids)
            return jsonify({"message": f"Deleted {n} transcript(s).", "count": n})
        return jsonify({"error": f"Unknown action {action!r}"}), 400

    @app.route("/download/<job_id>")
    def download(job_id):  # noqa: ANN202
        _settings, storage, jobs = _svc()
        job = storage.get_job(job_id)
        if job is None or job.status != "done":
            abort(404)
        kind = request.args.get("kind", "best")
        body, suffix, mime = _export(jobs, job, kind)
        if body is None:
            abort(404)
        return Response(body, mimetype=mime, headers={
            "Content-Disposition": _disposition(download_name(job, suffix))})

    @app.get("/download.zip")
    def download_zip():  # noqa: ANN202
        _settings, storage, jobs = _svc()
        ids = [i for i in request.args.get("ids", "").split(",") if i][:500]
        kind = request.args.get("kind", "best")
        buf = io.BytesIO()
        used = set()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for job in storage.get_jobs(ids):
                if job.status != "done":
                    continue
                body, suffix, _mime = _export(jobs, job, kind)
                if body is None:
                    continue
                name = download_name(job, suffix)
                if name in used:
                    name = download_name(job, f" (v{job.version or 1}){suffix}")
                used.add(name)
                z.writestr(name, body)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="transcripts.zip")

    # --- settings ----------------------------------------------------------------------------

    @app.route("/settings")
    def settings_page():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        from .supabase_sync import setup_sql
        store = current_app.config["CONNECTIONS"]
        public = store.public()
        # Shown exactly once, to the browser that created it.
        new_token = session.pop("new_token", None)
        return render_template(
            "settings.html", conn=public, setup_sql=setup_sql(public["supabase_table"] or "transcripts"),
            tokens=storage.list_tokens(), new_token=new_token, counts=storage.counts(),
            **_options_context(storage),
        )

    @app.post("/settings/connections")
    def save_connections():  # noqa: ANN202
        store = current_app.config["CONNECTIONS"]
        fields = ("openai_base_url", "openai_api_key", "openai_model", "ai_language",
                  "supabase_url", "supabase_key", "supabase_table")
        store.update({k: request.form.get(k) for k in fields if k in request.form})
        for key in ("openai_api_key", "supabase_key"):
            if request.form.get(f"clear_{key}"):
                store.clear(key)
        flash("Connections saved.")
        return redirect(url_for("settings_page") + "#connections")

    @app.post("/api/settings/test/<kind>")
    def test_connection(kind):  # noqa: ANN202
        conns = current_app.config["CONNECTIONS"].load()
        if kind == "openai":
            if not conns.ai_ready:
                return jsonify({"ok": False, "message": "Add a base URL, API key and model first."})
            try:
                finisher = current_app.config["AI_FACTORY"](conns)
                models = finisher.list_models()
            except Exception as e:  # noqa: BLE001
                return jsonify({"ok": False, "message": f"Failed: {e}"})
            ok = conns.openai_model in models or not models
            msg = (f"Connected — {len(models)} models available, '{conns.openai_model}' is one of them."
                   if ok else f"Connected, but model '{conns.openai_model}' is not in the list.")
            return jsonify({"ok": ok, "message": msg, "models": models[:200]})
        if kind == "supabase":
            if not conns.supabase_ready:
                return jsonify({"ok": False, "message": "Add the project URL, service key and table first."})
            from .supabase_sync import SupabaseConfig, check
            ok, msg = check(SupabaseConfig(conns.supabase_url, conns.supabase_key, conns.supabase_table))
            if ok:
                current_app.config["JOBS"]._sync_wake.set()
            return jsonify({"ok": ok, "message": msg})
        abort(404)

    @app.post("/settings/defaults")
    def save_defaults():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        preset = request.form.get("preset") or None
        opts = O.resolve(O.from_form(request.form), preset=preset)
        storage.set_setting("default_options", {k: v for k, v in opts.items() if k in O.BY_KEY})
        flash("Default settings saved — every device uses them for new uploads.")
        return redirect(url_for("settings_page") + "#defaults")

    @app.post("/api/settings/defaults")
    def save_defaults_json():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        body = request.get_json(silent=True) or {}
        opts = O.resolve(body.get("options") or {}, preset=body.get("preset"))
        storage.set_setting("default_options", {k: v for k, v in opts.items() if k in O.BY_KEY})
        return jsonify({"message": "Saved as the default for new uploads on every device."})

    @app.post("/settings/profiles")
    def save_profile():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        f = request.form
        if f.get("delete"):
            storage.delete_profile(int(f["delete"]))
            flash("Profile deleted.")
        else:
            try:
                storage.save_profile(f.get("name", ""), f.get("prompt", ""), f.get("hotwords", ""),
                                     f.get("languages", ""), f.get("notes", ""),
                                     int(f["id"]) if f.get("id", "").isdigit() else None)
                flash("Profile saved.")
            except Exception as e:  # noqa: BLE001
                flash(f"Could not save profile: {e}")
        return redirect(url_for("settings_page") + "#profiles")

    @app.post("/settings/tokens")
    def tokens():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        if request.form.get("delete"):
            storage.delete_token(int(request.form["delete"]))
            flash("Token revoked.")
            return redirect(url_for("settings_page") + "#tokens")
        name = (request.form.get("name") or "").strip()[:60] or "device"
        token = "vt_" + secrets.token_urlsafe(32)
        storage.add_token(name, hashlib.sha256(token.encode()).hexdigest(), token[:10])
        session["new_token"] = {"name": name, "token": token}
        return redirect(url_for("settings_page") + "#tokens")

    @app.get("/api/health")
    def health():  # noqa: ANN202
        _settings, storage, _jobs = _svc()
        from app.config import local_health_error
        import os
        url = os.getenv("LOCAL_WHISPER_URL") or ""
        gpu = None
        if url:
            err = local_health_error(url, timeout=2)
            gpu = {"ok": err is None, "message": err or "ready"}
        conns = current_app.config["CONNECTIONS"].load()
        return jsonify({"gpu": gpu, "counts": storage.counts(), "ai_ready": conns.ai_ready,
                        "supabase_ready": conns.supabase_ready})


def _disposition(name: str) -> str:
    from urllib.parse import quote
    ascii_name = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"


def _export(jobs, job, kind: str):
    from app.pipeline import to_srt, to_timestamped_text
    raw = jobs.read_text(job.transcript_path)
    clean = jobs.read_text(job.clean_text_path)
    if kind == "raw":
        return raw, ".txt", "text/plain; charset=utf-8"
    if kind == "clean":
        return clean, " (clean).txt", "text/plain; charset=utf-8"
    if kind in ("srt", "timestamps"):
        segs = jobs.read_segments(job)
        if segs is None:
            return None, "", ""
        if kind == "srt":
            return to_srt(segs), ".srt", "application/x-subrip; charset=utf-8"
        return to_timestamped_text(segs), " (timestamps).txt", "text/plain; charset=utf-8"
    if kind == "json":
        row = jobs.build_row(job)
        return json.dumps(row, ensure_ascii=False, indent=2), ".json", "application/json"
    # best: the cleaned text when there is one, with a short header
    text = clean or raw
    if text is None:
        return None, "", ""
    header = []
    if job.title:
        header.append(job.title)
    if job.description:
        header.append(job.description)
    body = ("\n\n".join(header) + "\n\n---\n\n" + text) if header else text
    return body, ".txt", "text/plain; charset=utf-8"
