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
from app.whisper_client import SegmentFilter, WhisperClient, filter_segments


def _status_error(cls, status):
    req = httpx.Request("POST", "https://api.test/v1/audio/transcriptions")
    resp = httpx.Response(status, request=req)
    return cls("err", response=resp, body=None)


def _make_client(monkeypatch, tmp_path, errors, language=None):
    """WhisperClient whose API raises the queued errors, then succeeds."""
    monkeypatch.setattr(wc_module.time, "sleep", lambda s: None)
    client = WhisperClient(api_key="k", language=language)
    calls = {"n": 0}

    def _fake_create_for(endpoint):
        def fake_create(**kwargs):
            calls["n"] += 1
            calls["endpoint"] = endpoint
            calls["kwargs"] = kwargs
            if errors:
                raise errors.pop(0)
            return SimpleNamespace(text="ok")

        return fake_create

    client.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=_fake_create_for("transcriptions")),
            translations=SimpleNamespace(create=_fake_create_for("translations")),
        )
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


def test_uses_transcriptions_endpoint_with_auto_detected_language(monkeypatch, tmp_path):
    # The transcriptions endpoint keeps the output in the spoken language
    # (Whisper auto-detects when no language is passed), which transcribes more
    # accurately than translating to English on the fly.
    # temperature=0 curbs Whisper's repetition/hallucination loops.
    client, calls, f = _make_client(monkeypatch, tmp_path, [])
    client.transcribe_file(f)
    assert calls["endpoint"] == "transcriptions"
    assert calls["kwargs"]["temperature"] == 0
    assert "language" not in calls["kwargs"]


def test_language_hint_is_passed_through(monkeypatch, tmp_path):
    # An explicit TRANSCRIBE_LANGUAGE hint improves accuracy when the language
    # is known up front; it must reach the API call.
    client, calls, f = _make_client(monkeypatch, tmp_path, [], language="ro")
    client.transcribe_file(f)
    assert calls["endpoint"] == "transcriptions"
    assert calls["kwargs"]["language"] == "ro"


# --- filter_segments (pure) ---------------------------------------------------

def test_filter_keeps_clean_segments_and_joins_them():
    segs = [
        {"text": " Hello there", "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.2},
        {"text": " how are you", "no_speech_prob": 0.1, "avg_logprob": -0.4, "compression_ratio": 1.3},
    ]
    text, dropped = filter_segments(segs, SegmentFilter())
    assert text == "Hello there how are you"
    assert dropped == []


def test_filter_drops_repetition_loop():
    segs = [
        {"text": " rast rast rast rast", "no_speech_prob": 0.0, "avg_logprob": -0.2, "compression_ratio": 3.5},
    ]
    text, dropped = filter_segments(segs, SegmentFilter())
    assert text == ""
    assert len(dropped) == 1 and "repetition" in dropped[0][1]


def test_filter_drops_no_speech_only_when_both_conditions_hold():
    segs = [
        # High no_speech AND low confidence -> silence, drop.
        {"text": " uhh", "no_speech_prob": 0.9, "avg_logprob": -2.0, "compression_ratio": 1.0},
        # High no_speech but confident -> real speech, keep.
        {"text": " keep me", "no_speech_prob": 0.9, "avg_logprob": -0.1, "compression_ratio": 1.0},
    ]
    text, dropped = filter_segments(segs, SegmentFilter())
    assert text == "keep me"
    assert len(dropped) == 1 and "no speech" in dropped[0][1]


def test_filter_works_on_object_segments():
    segs = [
        SimpleNamespace(text=" object seg", no_speech_prob=0.0, avg_logprob=-0.2, compression_ratio=1.1),
    ]
    text, _ = filter_segments(segs, SegmentFilter())
    assert text == "object seg"


# --- verbose vs text response handling ---------------------------------------

def _client_with_capture(model, segment_filter):
    """A WhisperClient whose create() records kwargs and returns a canned resp."""
    captured: dict = {}

    def make_resp():
        return SimpleNamespace(
            text="fallback text",
            segments=[
                SimpleNamespace(text=" good", no_speech_prob=0.0, avg_logprob=-0.2, compression_ratio=1.2),
                SimpleNamespace(text=" loop", no_speech_prob=0.0, avg_logprob=-0.2, compression_ratio=3.9),
            ],
        )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return make_resp()

    client = WhisperClient(api_key="k", model=model, segment_filter=segment_filter)
    client.client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=fake_create))
    )
    return client, captured


def test_whisper_model_uses_verbose_json_and_filters(tmp_path):
    client, captured = _client_with_capture("whisper-large-v3", SegmentFilter())
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    out = client.transcribe_file(f)
    assert captured["response_format"] == "verbose_json"
    assert out == "good"  # the high-compression "loop" segment is filtered out


def test_gpt4o_model_stays_on_text_even_with_filter(tmp_path):
    # gpt-4o-transcribe cannot return verbose_json; requesting it would 400.
    client, captured = _client_with_capture("gpt-4o-transcribe", SegmentFilter())
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    out = client.transcribe_file(f)
    assert captured["response_format"] == "text"
    assert out == "fallback text"


def test_no_filter_configured_uses_text_mode(tmp_path):
    client, captured = _client_with_capture("whisper-large-v3", None)
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    out = client.transcribe_file(f)
    assert captured["response_format"] == "text"
    assert out == "fallback text"
