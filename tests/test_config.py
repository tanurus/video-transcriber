import pytest

from app.config import Config, GROQ_BASE_URL


@pytest.fixture
def env_file(tmp_path):
    """An empty dotenv file so Config.load never reads the repo's real .env."""
    p = tmp_path / "test.env"
    p.write_text("", encoding="utf-8")
    return str(p)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "TRANSCRIBE_MODEL",
        "TRANSCRIBE_LANGUAGE",
        "CANDIDATE_LANGUAGES",
        "OPENAI_TIMEOUT",
        "CHUNK_TARGET_MB",
        "AUDIO_BITRATE",
        "CHUNK_TARGET_SEC",
        "CHUNK_MAX_SEC",
        "SILENCE_NOISE_DB",
        "SILENCE_MIN_SEC",
        "MAX_CONCURRENCY",
        "NO_SPEECH_THRESHOLD",
        "LOGPROB_THRESHOLD",
        "COMPRESSION_RATIO_THRESHOLD",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_blank_optional_env_vars_fall_back_to_defaults(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Blank placeholders (e.g. "OPENAI_TIMEOUT=") must behave like unset vars.
    monkeypatch.setenv("TRANSCRIBE_MODEL", "")
    monkeypatch.setenv("OPENAI_TIMEOUT", "")
    monkeypatch.setenv("CHUNK_TARGET_MB", "")
    monkeypatch.setenv("AUDIO_BITRATE", "")

    cfg = Config.load(env_path=env_file)

    assert cfg.model == "gpt-4o-transcribe"
    assert cfg.timeout == 600
    assert cfg.chunk_target_mb == 24
    assert cfg.audio_bitrate == "96k"


def test_language_defaults_to_auto_detect(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.language is None


def test_language_env_override(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE", "de")
    cfg = Config.load(env_path=env_file)
    assert cfg.language == "de"


def test_blank_language_falls_back_to_auto_detect(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE", "")
    cfg = Config.load(env_path=env_file)
    assert cfg.language is None


def test_groq_key_takes_precedence(monkeypatch, env_file):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.openai_api_key == "gsk-test"
    assert cfg.base_url == GROQ_BASE_URL
    # whisper-large-v3 (not turbo) is Groq's highest-accuracy Whisper model.
    assert cfg.model == "whisper-large-v3"


def test_candidate_languages_default_none(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.candidate_languages is None


def test_candidate_languages_parsed_from_csv(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CANDIDATE_LANGUAGES", "ro, ru , en")
    cfg = Config.load(env_path=env_file)
    assert cfg.candidate_languages == ["ro", "ru", "en"]


def test_chunking_and_filter_defaults(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.chunk_target_sec == 25
    assert cfg.chunk_max_sec == 40
    assert cfg.silence_noise_db == "-30dB"
    assert cfg.silence_min_sec == 0.5
    assert cfg.max_concurrency == 4
    assert cfg.no_speech_threshold == 0.6
    assert cfg.logprob_threshold == -1.0
    assert cfg.compression_ratio_threshold == 2.4


def test_chunking_and_filter_env_overrides(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHUNK_TARGET_SEC", "30")
    monkeypatch.setenv("CHUNK_MAX_SEC", "60")
    monkeypatch.setenv("SILENCE_NOISE_DB", "-25dB")
    monkeypatch.setenv("SILENCE_MIN_SEC", "0.3")
    monkeypatch.setenv("MAX_CONCURRENCY", "8")
    monkeypatch.setenv("COMPRESSION_RATIO_THRESHOLD", "2.0")
    cfg = Config.load(env_path=env_file)
    assert cfg.chunk_target_sec == 30
    assert cfg.chunk_max_sec == 60
    assert cfg.silence_noise_db == "-25dB"
    assert cfg.silence_min_sec == 0.3
    assert cfg.max_concurrency == 8
    assert cfg.compression_ratio_threshold == 2.0


def test_max_concurrency_has_a_floor_of_one(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_CONCURRENCY", "0")
    cfg = Config.load(env_path=env_file)
    assert cfg.max_concurrency == 1


def test_blank_new_env_vars_fall_back_to_defaults(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for var in ("CHUNK_TARGET_SEC", "SILENCE_MIN_SEC", "MAX_CONCURRENCY", "NO_SPEECH_THRESHOLD"):
        monkeypatch.setenv(var, "")
    cfg = Config.load(env_path=env_file)
    assert cfg.chunk_target_sec == 25
    assert cfg.silence_min_sec == 0.5
    assert cfg.max_concurrency == 4
    assert cfg.no_speech_threshold == 0.6
