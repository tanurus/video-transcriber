import io
from types import SimpleNamespace

import pytest

from gpu_server.model_manager import ModelManager, UnknownModelError, normalize_model
from gpu_server.server import create_app


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel: records calls, returns fixed segments."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        segs = [
            SimpleNamespace(
                id=1, seek=0, start=0.0, end=2.5, text=" Hello there.", tokens=[1, 2],
                temperature=0.0, avg_logprob=-0.12, compression_ratio=1.1, no_speech_prob=0.01,
            ),
            SimpleNamespace(
                id=2, seek=250, start=2.5, end=4.0, text=" Bună ziua.", tokens=[3],
                temperature=0.2, avg_logprob=-0.3, compression_ratio=1.0, no_speech_prob=0.02,
            ),
        ]
        info = SimpleNamespace(language="ro", language_probability=0.9, duration=4.0)
        return iter(segs), info


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def loads():
    return []


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def manager(loads, clock):
    def loader(name):
        loads.append(name)
        return FakeWhisperModel(name)

    return ModelManager(
        default_model="large-v3",
        allowed_models=["large-v3", "large-v3-turbo"],
        idle_unload_sec=900,
        loader=loader,
        clock=clock,
    )


@pytest.fixture
def client(manager):
    app = create_app(manager=manager, cuda_probe=lambda: 1, start_background=False)
    return app.test_client()


def _post(client, **form):
    data = {"file": (io.BytesIO(b"fake-audio"), "chunk.mp3"), "model": "whisper-large-v3"}
    data.update(form)
    return client.post("/v1/audio/transcriptions", data=data, content_type="multipart/form-data")


# --- model name handling -----------------------------------------------------

@pytest.mark.parametrize(
    "requested,expected",
    [
        ("whisper-large-v3", "large-v3"),
        ("large-v3", "large-v3"),
        ("whisper-large-v3-turbo", "large-v3-turbo"),
        ("Whisper-Large-V3", "large-v3"),
        ("whisper-1", "large-v3"),  # OpenAI's generic id maps to the default
        ("", "large-v3"),
    ],
)
def test_normalize_model(requested, expected):
    assert normalize_model(requested, "large-v3", ["large-v3", "large-v3-turbo"]) == expected


def test_normalize_rejects_models_outside_allowlist():
    # An arbitrary id must never trigger a multi-GB download.
    with pytest.raises(UnknownModelError):
        normalize_model("whisper-tiny", "large-v3", ["large-v3"])


# --- /v1/audio/transcriptions ------------------------------------------------

def test_verbose_json_carries_segment_confidence_stats(client):
    # The app's hallucination filter and language race read these three fields;
    # if they were missing it would silently default them to 0.0 and switch off.
    resp = _post(client, response_format="verbose_json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["language"] == "ro"
    assert body["duration"] == 4.0
    assert body["text"] == "Hello there. Bună ziua."
    seg = body["segments"][0]
    for key in ("avg_logprob", "no_speech_prob", "compression_ratio", "start", "end", "text"):
        assert key in seg
    assert seg["avg_logprob"] == pytest.approx(-0.12)


def test_json_format_returns_text_only(client):
    resp = _post(client, response_format="json")
    assert resp.get_json() == {"text": "Hello there. Bună ziua."}


def test_text_format_returns_plain_text(client):
    resp = _post(client, response_format="text")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    assert resp.get_data(as_text=True) == "Hello there. Bună ziua."


def test_unsupported_format_is_400(client):
    assert _post(client, response_format="srt").status_code == 400


def test_missing_file_is_400(client):
    resp = client.post("/v1/audio/transcriptions", data={"model": "whisper-large-v3"})
    assert resp.status_code == 400


def test_unknown_model_is_400(client):
    assert _post(client, model="whisper-tiny").status_code == 400


def test_language_and_prompt_are_passed_through(client, manager):
    _post(client, language="ru", prompt="Kayak, Skyscanner")
    model = manager._model
    _, kwargs = model.calls[-1]
    assert kwargs["language"] == "ru"
    assert kwargs["initial_prompt"] == "Kayak, Skyscanner"


def test_temperature_zero_keeps_whisper_fallback_schedule(client, manager):
    # OpenAI semantics: temperature=0 means "start at 0, raise it on a failed decode".
    _post(client, temperature="0")
    _, kwargs = manager._model.calls[-1]
    assert kwargs["temperature"][0] == 0.0
    assert len(kwargs["temperature"]) > 1


def test_nonzero_temperature_is_used_as_is(client, manager):
    _post(client, temperature="0.4")
    _, kwargs = manager._model.calls[-1]
    assert kwargs["temperature"] == [0.4]


def test_temp_upload_is_deleted_after_transcription(client, manager):
    _post(client)
    path, _ = manager._model.calls[-1]
    import os

    assert not os.path.exists(path)
    assert path.endswith(".mp3"), "PyAV sniffs the container, keep the extension"


def test_bearer_key_enforced_when_configured(manager):
    app = create_app(manager=manager, cuda_probe=lambda: 1, api_key="s3cret", start_background=False)
    c = app.test_client()
    assert _post(c).status_code == 401
    data = {"file": (io.BytesIO(b"x"), "a.mp3"), "model": "whisper-large-v3"}
    ok = c.post(
        "/v1/audio/transcriptions", data=data, content_type="multipart/form-data",
        headers={"Authorization": "Bearer s3cret"},
    )
    assert ok.status_code == 200


def test_transcription_failure_is_500_json(manager):
    def broken_loader(name):
        raise RuntimeError("CUDA driver missing")

    manager._loader = broken_loader
    app = create_app(manager=manager, cuda_probe=lambda: 1, start_background=False)
    resp = _post(app.test_client())
    assert resp.status_code == 500
    assert "CUDA driver missing" in resp.get_json()["error"]["message"]


# --- model lifecycle ---------------------------------------------------------

def test_model_loads_lazily_and_is_reused(client, loads):
    assert loads == []
    _post(client)
    _post(client)
    assert loads == ["large-v3"]


def test_switching_model_reloads(client, loads):
    _post(client, model="whisper-large-v3")
    _post(client, model="whisper-large-v3-turbo")
    assert loads == ["large-v3", "large-v3-turbo"]


def test_idle_model_is_unloaded_to_free_vram(client, manager, clock):
    _post(client)
    clock.now += 899
    assert manager.maybe_unload() is False
    clock.now += 2
    assert manager.maybe_unload() is True
    assert manager.loaded_model is None


# --- /health and /v1/models --------------------------------------------------

def test_health_ok_without_loading_model(client, loads):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["cuda_devices"] == 1
    assert body["loaded_model"] is None
    assert loads == [], "a health probe must never pull 3 GB into VRAM"


def test_health_503_when_gpu_missing(manager):
    # After a kernel update the NVIDIA driver can vanish; health must say so.
    app = create_app(manager=manager, cuda_probe=lambda: 0, start_background=False)
    resp = app.test_client().get("/health")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "no-gpu"


def test_models_lists_openai_style_ids(client):
    ids = [m["id"] for m in client.get("/v1/models").get_json()["data"]]
    assert ids == ["whisper-large-v3", "whisper-large-v3-turbo"]


# --- native decoding options ----------------------------------------------------

def test_decode_options_are_validated_clamped_and_passed(client, manager):
    _post(client, beam_size="99", best_of="3", patience="1.5", condition_on_previous_text="true",
          repetition_penalty="1.1", no_repeat_ngram_size="3", hotwords="AranGrant, Kayak",
          temperature_fallback="false", junk_field="rm -rf")
    _, kw = manager._model.calls[-1]
    assert kw["beam_size"] == 10 and kw["best_of"] == 3 and kw["patience"] == 1.5
    assert kw["condition_on_previous_text"] is True and kw["no_repeat_ngram_size"] == 3
    assert kw["hotwords"] == "AranGrant, Kayak"
    assert kw["temperature"] == [0.0]  # fallback switched off
    assert "junk_field" not in kw and kw["vad_filter"] is False


def test_native_mode_passes_vad_parameters(client, manager):
    _post(client, vad_filter="true", vad_threshold="0.35", vad_min_silence_ms="800")
    _, kw = manager._model.calls[-1]
    assert kw["vad_filter"] is True
    assert kw["vad_parameters"] == {"threshold": 0.35, "min_silence_duration_ms": 800,
                                    "speech_pad_ms": 300, "min_speech_duration_ms": 250}


def test_hallucination_silence_turns_on_word_timestamps(client, manager):
    _post(client, hallucination_silence_threshold="2")
    _, kw = manager._model.calls[-1]
    assert kw["hallucination_silence_threshold"] == 2.0 and kw["word_timestamps"] is True


def test_bad_number_is_400(client):
    assert _post(client, beam_size="five").status_code == 400
