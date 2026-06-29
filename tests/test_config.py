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
        "OPENAI_TIMEOUT",
        "CHUNK_TARGET_MB",
        "AUDIO_BITRATE",
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


def test_language_defaults_to_english(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.language == "en"


def test_language_env_override(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE", "de")
    cfg = Config.load(env_path=env_file)
    assert cfg.language == "de"


def test_blank_language_falls_back_to_english(monkeypatch, env_file):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE", "")
    cfg = Config.load(env_path=env_file)
    assert cfg.language == "en"


def test_groq_key_takes_precedence(monkeypatch, env_file):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.openai_api_key == "gsk-test"
    assert cfg.base_url == GROQ_BASE_URL
    assert cfg.model == "whisper-large-v3-turbo"
