"""Chunk-level API for the n8n validated pipeline (token-authenticated).

    POST /api/v2/sessions                                 {job_id, ...settings, allowed_languages}
    GET  /api/v2/sessions/<sid>[?attempts=1]
    POST /api/v2/sessions/<sid>/chunks/<cid>/transcribe   {...settings, label}
    GET  /api/v2/sessions/<sid>/chunks/<cid>/attempts
    POST /api/v2/sessions/<sid>/chunks/<cid>/probe        {attempt_id?}  per-segment language re-detection
    POST /api/v2/sessions/<sid>/chunks/<cid>/recut        {start, end}
    POST /api/v2/sessions/<sid>/chunks/<cid>/split        {at}
    POST /api/v2/sessions/<sid>/attempts/<aid>/validation {passed, ...anything}
    POST /api/v2/sessions/<sid>/assemble                  {choices: {cid: aid}, fallback, report}
"""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from .auth import require_token
from .sessions import SessionError

bp = Blueprint("api_v2", __name__, url_prefix="/api/v2")


@bp.before_request
def _auth():  # noqa: ANN202
    return require_token()


@bp.errorhandler(SessionError)
def _session_error(e):  # noqa: ANN202
    return jsonify({"error": str(e)}), e.status


def _svc():
    return current_app.config["SESSIONS"]


def _body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@bp.post("/sessions")
def create_session():  # noqa: ANN202
    body = _body()
    job_id = str(body.pop("job_id", "") or "")
    if not job_id:
        return jsonify({"error": "job_id is required."}), 400
    return jsonify(_svc().create(job_id, body)), 201


@bp.get("/sessions/<sid>")
def get_session(sid):  # noqa: ANN202
    return jsonify(_svc().get(sid, include_attempts=request.args.get("attempts") == "1"))


@bp.post("/sessions/<sid>/chunks/<int:cid>/transcribe")
def transcribe_chunk(sid, cid):  # noqa: ANN202
    body = _body()
    label = body.pop("label", None)
    return jsonify(_svc().transcribe(sid, cid, body, label=label))


@bp.post("/sessions/<sid>/chunks/<int:cid>/probe")
def probe_chunk(sid, cid):  # noqa: ANN202
    return jsonify(_svc().probe(sid, cid, _body().get("attempt_id")))


@bp.get("/sessions/<sid>/chunks/<int:cid>/attempts")
def chunk_attempts(sid, cid):  # noqa: ANN202
    return jsonify({"attempts": _svc().attempts(sid, cid)})


@bp.post("/sessions/<sid>/chunks/<int:cid>/recut")
def recut_chunk(sid, cid):  # noqa: ANN202
    body = _body()
    try:
        return jsonify(_svc().recut(sid, cid, float(body["start"]), float(body["end"])))
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, SessionError):
            raise
        return jsonify({"error": "start and end (seconds) are required."}), 400


@bp.post("/sessions/<sid>/chunks/<int:cid>/split")
def split_chunk(sid, cid):  # noqa: ANN202
    at = _body().get("at")
    return jsonify(_svc().split(sid, cid, float(at) if at is not None else None))


@bp.post("/sessions/<sid>/attempts/<aid>/validation")
def record_validation(sid, aid):  # noqa: ANN202
    return jsonify(_svc().record_validation(sid, aid, _body()))


@bp.post("/sessions/<sid>/assemble")
def assemble(sid):  # noqa: ANN202
    body = _body()
    fallback = body.get("fallback", "best")
    if fallback not in ("best", "error"):
        return jsonify({"error": "fallback must be 'best' or 'error'."}), 400
    return jsonify(_svc().assemble(sid, body.get("choices") or {}, fallback, body.get("report")))
