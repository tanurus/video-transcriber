from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

import app.whisper_client as wc_module
from app.whisper_client import WhisperClient


def _status_error(cls, status):
    req = httpx.Request("POST", "https://api.test/v1/audio/transcriptions")
    resp = httpx.Response(status, request=req)
    return cls("err", response=resp, body=None)


def _make_client(monkeypatch, tmp_path, errors):
    """WhisperClient whose API raises the queued errors, then succeeds."""
    monkeypatch.setattr(wc_module.time, "sleep", lambda s: None)
    client = WhisperClient(api_key="k")
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        calls["kwargs"] = kwargs
        if errors:
            raise errors.pop(0)
        return SimpleNamespace(text="ok")

    client.client = SimpleNamespace(
        audio=SimpleNamespace(translations=SimpleNamespace(create=fake_create))
    )
    audio_file = tmp_path / "a.mp3"
    audio_file.write_bytes(b"x")
    return client, calls, audio_file


def test_auth_error_not_retried(monkeypatch, tmp_path):
    err = _status_error(AuthenticationError, 401)
    client, calls, f = _make_client(monkeypatch, tmp_path, [err, err, err])
    with pytest.raises(AuthenticationError):
        client.transcribe_file(f)
    assert calls["n"] == 1, "401 is not transient; retrying re-uploads the file pointlessly"


def test_bad_request_not_retried(monkeypatch, tmp_path):
    err = _status_error(BadRequestError, 400)
    client, calls, f = _make_client(monkeypatch, tmp_path, [err, err, err])
    with pytest.raises(BadRequestError):
        client.transcribe_file(f)
    assert calls["n"] == 1


def test_rate_limit_retried(monkeypatch, tmp_path):
    errors = [_status_error(RateLimitError, 429), _status_error(RateLimitError, 429)]
    client, calls, f = _make_client(monkeypatch, tmp_path, errors)
    assert client.transcribe_file(f) == "ok"
    assert calls["n"] == 3


def test_server_error_retried(monkeypatch, tmp_path):
    errors = [_status_error(InternalServerError, 500)]
    client, calls, f = _make_client(monkeypatch, tmp_path, errors)
    assert client.transcribe_file(f) == "ok"
    assert calls["n"] == 2


def test_connection_error_retried(monkeypatch, tmp_path):
    req = httpx.Request("POST", "https://api.test/v1/audio/transcriptions")
    errors = [APIConnectionError(request=req)]
    client, calls, f = _make_client(monkeypatch, tmp_path, errors)
    assert client.transcribe_file(f) == "ok"
    assert calls["n"] == 2


def test_uses_translations_endpoint_at_temperature_zero(monkeypatch, tmp_path):
    # The translations endpoint forces English output regardless of the spoken
    # language; temperature=0 curbs Whisper's repetition/hallucination loops.
    client, calls, f = _make_client(monkeypatch, tmp_path, [])
    client.transcribe_file(f)
    assert calls["kwargs"]["temperature"] == 0
    assert "language" not in calls["kwargs"]
