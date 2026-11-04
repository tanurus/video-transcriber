"""OpenAI-compatible Whisper transcription server backed by a local GPU.

Implements just enough of ``POST /v1/audio/transcriptions`` for the OpenAI SDK
(and therefore ``app.whisper_client``) to use it as a drop-in base_url.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, Response, jsonify, request

from .model_manager import ModelManager, UnknownModelError, parse_decode_options

SUPPORTED_FORMATS = {"json", "text", "verbose_json"}


def _error(message: str, status: int, kind: str = "invalid_request_error"):
    return jsonify({"error": {"message": message, "type": kind}}), status


def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - no driver / no library both mean "no GPU"
        return 0


def manager_from_env() -> ModelManager:
    allowed = [
        m.strip() for m in (os.getenv("WHISPER_ALLOWED_MODELS") or "large-v3,large-v3-turbo").split(",")
        if m.strip()
    ]
    return ModelManager(
        default_model=os.getenv("WHISPER_MODEL") or "large-v3",
        allowed_models=allowed,
        device=os.getenv("WHISPER_DEVICE") or "cuda",
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE") or "float16",
        idle_unload_sec=int(os.getenv("WHISPER_IDLE_UNLOAD_SEC") or "900"),
        beam_size=int(os.getenv("WHISPER_BEAM_SIZE") or "5"),
    )


def _start_unload_thread(manager: ModelManager, interval: int = 30) -> None:
    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                manager.maybe_unload()
            except Exception:  # noqa: BLE001 - the sweeper must never die
                pass

    threading.Thread(target=loop, name="whisper-idle-unload", daemon=True).start()


def create_app(
    manager: Optional[ModelManager] = None,
    cuda_probe: Callable[[], int] = _cuda_device_count,
    api_key: Optional[str] = None,
    start_background: bool = True,
) -> Flask:
    manager = manager or manager_from_env()
    if api_key is None:
        api_key = os.getenv("WHISPER_API_KEY") or None

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("WHISPER_MAX_UPLOAD_MB") or "1024") * 1024 * 1024

    @app.before_request
    def _auth():  # noqa: ANN202
        # The tailnet is the access boundary; a key is optional defence in depth.
        if api_key and request.path.startswith("/v1/"):
            if request.headers.get("Authorization", "") != f"Bearer {api_key}":
                return _error("Invalid API key.", 401, "authentication_error")
        return None

    @app.get("/health")
    def health():  # noqa: ANN202
        devices = cuda_probe()
        body = {
            "status": "ok" if devices > 0 else "no-gpu",
            "cuda_devices": devices,
            "default_model": manager.default_model,
            "loaded_model": manager.loaded_model,
            "busy": manager.busy,
        }
        return jsonify(body), (200 if devices > 0 else 503)

    @app.get("/v1/models")
    def models():  # noqa: ANN202
        return jsonify(
            {
                "object": "list",
                "data": [
                    {"id": f"whisper-{m}", "object": "model", "owned_by": "local"}
                    for m in manager.allowed_models
                ],
            }
        )

    @app.post("/v1/audio/transcriptions")
    def transcriptions():  # noqa: ANN202
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return _error("Missing 'file' upload.", 400)
        fmt = (request.form.get("response_format") or "json").lower()
        if fmt not in SUPPORTED_FORMATS:
            return _error(f"response_format '{fmt}' is not supported; use json, text or verbose_json.", 400)
        try:
            temperature = float(request.form.get("temperature") or 0)
        except ValueError:
            return _error("temperature must be a number.", 400)
        try:
            manager.resolve(request.form.get("model") or "")
            decode = parse_decode_options(request.form)
        except (UnknownModelError, ValueError) as e:
            return _error(str(e), 400)

        # Keep the extension: the decoder sniffs the container from it.
        suffix = Path(upload.filename).suffix or ".bin"
        fd, tmp = tempfile.mkstemp(prefix="whisper_", suffix=suffix)
        os.close(fd)
        try:
            upload.save(tmp)
            result = manager.transcribe(
                tmp,
                model=request.form.get("model") or "",
                language=request.form.get("language") or None,
                temperature=temperature,
                prompt=request.form.get("prompt") or None,
                decode=decode,
            )
        except Exception as e:  # noqa: BLE001 - surface as an API error, not a stack trace
            return _error(f"Transcription failed: {e}", 500, "server_error")
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        if fmt == "text":
            return Response(result["text"], mimetype="text/plain")
        if fmt == "json":
            return jsonify({"text": result["text"]})
        return jsonify(result)

    if start_background:
        _start_unload_thread(manager)

    return app
