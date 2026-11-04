"""Token-authenticated configuration API, used by the MCP server (``mcp_server/``).

Everything the Settings page can change about *transcription* is here: the option
catalogue, the defaults every device uses, and profiles. Connection secrets
(OpenAI / Supabase keys) and API tokens are deliberately absent: a token can read
whether a connection is ready, never change a key or mint another token.

    GET    /api/v1/settings/catalog      every option with type, range, choices, help
    GET    /api/v1/settings/defaults     the defaults new uploads start from
    PUT    /api/v1/settings/defaults     {"preset": "...", "options": {...}, "merge": true}
    GET    /api/v1/profiles
    POST   /api/v1/profiles              {"name", "prompt", "hotwords", "languages", "notes", "id"?}
    DELETE /api/v1/profiles/<id>
    GET    /api/v1/connections           readiness + non-secret fields only
    GET    /api/v1/health                GPU, library counts, readiness
    GET    /api/v1/jobs/<id>/segments    timestamped segments
    GET    /api/v1/jobs/<id>/versions    every regeneration of the same recording
    POST   /api/v1/jobs/actions          {"action": "regenerate"|"ai"|"sync", "ids": [...], ...}
"""
from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request

from app import options as O

from .auth import require_token
from .routes import health_payload, job_json, run_bulk

bp = Blueprint("api_config", __name__, url_prefix="/api/v1")

# Non-secret connection fields a token may read.
PUBLIC_CONNECTION_FIELDS = ("openai_base_url", "openai_model", "ai_language", "supabase_url", "supabase_table")


@bp.before_request
def _auth():  # noqa: ANN202
    return require_token()


def _storage():
    return current_app.config["STORAGE"]


def _stored_defaults() -> dict:
    return _storage().get_setting("default_options", {}) or {}


def _profile_json(p) -> dict:
    return asdict(p)


@bp.get("/settings/catalog")
def catalog():  # noqa: ANN202
    profiles = _storage().list_profiles()
    options = []
    for o in O.CATALOG:
        item = asdict(o)
        item["choices"] = [{"value": v, "label": label} for v, label in o.choices]
        if o.key == "profile_id":
            item["choices"] = [{"value": "", "label": "None"}] + [
                {"value": str(p.id), "label": p.name} for p in profiles
            ]
        item["show_if"] = {"key": o.show_if[0], "values": list(o.show_if[1])} if o.show_if else None
        options.append(item)
    return jsonify({
        "groups": [{"key": k, "label": label} for k, label in O.GROUPS],
        "options": options,
        "presets": [{"key": k, "label": label, "overrides": O.PRESETS[k]} for k, label in O.PRESET_LABELS],
    })


@bp.get("/settings/defaults")
def get_defaults():  # noqa: ANN202
    resolved = O.resolve(base=_stored_defaults())
    return jsonify({"options": resolved, "summary": O.summary(resolved)})


@bp.put("/settings/defaults")
def put_defaults():  # noqa: ANN202
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object."}), 400
    raw = body.get("options") or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "'options' must be an object."}), 400
    unknown = sorted(k for k in raw if k not in O.BY_KEY)
    if unknown:
        return jsonify({"error": f"Unknown option(s): {', '.join(unknown)}. See /api/v1/settings/catalog."}), 400
    preset = body.get("preset") or None
    if preset and preset not in O.PRESETS:
        return jsonify({"error": f"Unknown preset {preset!r}. Use one of: {', '.join(O.PRESETS)}."}), 400
    # merge (default): change only the keys sent, on top of the current defaults.
    base = _stored_defaults() if body.get("merge", True) else None
    opts = O.resolve(raw, base=base, preset=preset)
    stored = {k: v for k, v in opts.items() if k in O.BY_KEY or k == "preset"}
    _storage().set_setting("default_options", stored)
    return jsonify({"message": "Saved as the default for new uploads on every device.",
                    "options": stored, "summary": O.summary(opts)})


@bp.get("/profiles")
def list_profiles():  # noqa: ANN202
    return jsonify({"profiles": [_profile_json(p) for p in _storage().list_profiles()]})


@bp.post("/profiles")
def save_profile():  # noqa: ANN202
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object."}), 400
    storage = _storage()
    pid = body.get("id")
    existing = None
    if pid not in (None, ""):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return jsonify({"error": "'id' must be a number."}), 400
        existing = storage.get_profile(pid)
        if existing is None:
            return jsonify({"error": f"No profile {pid}."}), 404
    # On update, fields not sent keep their current value.
    def field(name: str, limit: int) -> str:
        if name in body and body[name] is not None:
            return str(body[name]).strip()[:limit]
        return getattr(existing, name, "") if existing else ""
    try:
        new_id = storage.save_profile(field("name", 80), field("prompt", 4000), field("hotwords", 2000),
                                      field("languages", 100), field("notes", 2000), pid if existing else None)
    except Exception as e:  # noqa: BLE001  (empty name, duplicate name)
        return jsonify({"error": f"Could not save profile: {e}"}), 400
    return jsonify({"profile": _profile_json(storage.get_profile(new_id))}), (200 if existing else 201)


@bp.delete("/profiles/<int:profile_id>")
def delete_profile(profile_id):  # noqa: ANN202
    storage = _storage()
    if storage.get_profile(profile_id) is None:
        return jsonify({"error": f"No profile {profile_id}."}), 404
    storage.delete_profile(profile_id)
    return jsonify({"message": f"Profile {profile_id} deleted."})


@bp.get("/connections")
def connections():  # noqa: ANN202
    conns = current_app.config["CONNECTIONS"].load()
    data = {k: getattr(conns, k) for k in PUBLIC_CONNECTION_FIELDS}
    data.update({"ai_ready": conns.ai_ready, "supabase_ready": conns.supabase_ready,
                 "note": "Keys are entered on the web Settings page only."})
    return jsonify(data)


@bp.get("/health")
def health():  # noqa: ANN202
    return jsonify(health_payload())


@bp.get("/jobs/<job_id>/segments")
def segments(job_id):  # noqa: ANN202
    job = _storage().get_job(job_id)
    if job is None:
        return jsonify({"error": "not found"}), 404
    if job.status != "done":
        return jsonify({"error": f"Job is {job.status}, not done."}), 409
    return jsonify({"id": job.id, "segments": current_app.config["JOBS"].read_segments(job) or []})


@bp.get("/jobs/<job_id>/versions")
def versions(job_id):  # noqa: ANN202
    storage = _storage()
    if storage.get_job(job_id) is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"versions": [job_json(j) | {"settings": j.settings} for j in storage.versions_of(job_id)]})


@bp.post("/jobs/actions")
def job_actions():  # noqa: ANN202
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object."}), 400
    # No delete over the token API: removing recordings stays a deliberate click in the library.
    payload, status = run_bulk(current_app.config["JOBS"], current_app.config["CONNECTIONS"].load(), body,
                               allowed=("regenerate", "ai", "sync"))
    return jsonify(payload), status
