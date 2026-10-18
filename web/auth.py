"""Bearer-token check shared by the machine APIs (/api/v1, /api/v2)."""
from __future__ import annotations

import hashlib
from typing import Optional

from flask import current_app, jsonify, request


def require_token() -> Optional[tuple]:
    """Return an error response, or None after recording the token's name on the request."""
    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else request.args.get("token", "")
    if not token:
        return jsonify({"error": "Missing API token (Settings → Other devices & API)."}), 401
    record = current_app.config["STORAGE"].token_by_hash(hashlib.sha256(token.encode()).hexdigest())
    if record is None:
        return jsonify({"error": "Invalid or revoked API token."}), 401
    request.environ["vt.token_name"] = record.name
    return None
