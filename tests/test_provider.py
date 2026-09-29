import pytest

import app.config as config_module
from app.config import DEFAULT_LOCAL_URL, GROQ_BASE_URL, Config


@pytest.fixture
def env_file(tmp_path):
    p = tmp_path / "test.env"
    p.write_text("", encoding="utf-8")
    return str(p)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "GROQ_API_KEY", "OPENAI_API_KEY", "TRANSCRIBE_MODEL",
        "TRANSCRIBE_PROVIDER", "LOCAL_WHISPER_URL", "LOCAL_WHISPER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def health(monkeypatch):
    """Control what the local /health probe reports, and record the URLs probed."""
    state = {"ok": True, "probed": []}

    def fake(url, timeout=2.0):
        state["probed"].append(url)
        return None if state["ok"] else "connection refused"

    monkeypatch.setattr(config_module, "local_health_error", fake)
    return state


def test_auto_without_local_url_keeps_cloud_behaviour(monkeypatch, env_file, health):
    # Existing desktop installs have only a Groq key; nothing may change for them.
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.provider == "groq"
    assert cfg.base_url == GROQ_BASE_URL
    assert health["probed"] == [], "no local URL configured -> no network probe"


def test_auto_prefers_healthy_local_server(monkeypatch, env_file, health):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LOCAL_WHISPER_URL", "http://gpu.example:18921/v1")
    cfg = Config.load(env_path=env_file)
    assert cfg.provider == "local"
    assert cfg.base_url == "http://gpu.example:18921/v1"
    assert cfg.model == "whisper-large-v3"


def test_auto_falls_back_to_cloud_when_local_is_down(monkeypatch, env_file, health):
    health["ok"] = False
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LOCAL_WHISPER_URL", "http://gpu.example:18921/v1")
    cfg = Config.load(env_path=env_file)
    assert cfg.provider == "groq"


def test_auto_local_down_and_no_key_raises_clear_error(monkeypatch, env_file, health):
    health["ok"] = False
    monkeypatch.setenv("LOCAL_WHISPER_URL", "http://gpu.example:18921/v1")
    with pytest.raises(RuntimeError, match="gpu.example:18921.*not responding"):
        Config.load(env_path=env_file)


def test_explicit_local_needs_no_api_key(monkeypatch, env_file, health):
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "local")
    cfg = Config.load(env_path=env_file)
    assert cfg.provider == "local"
    assert cfg.base_url == DEFAULT_LOCAL_URL
    assert cfg.openai_api_key  # the SDK refuses an empty key; any placeholder works


def test_explicit_local_down_raises_clear_error(monkeypatch, env_file, health):
    health["ok"] = False
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "local")
    with pytest.raises(RuntimeError, match="not responding"):
        Config.load(env_path=env_file)


def test_local_api_key_is_forwarded(monkeypatch, env_file, health):
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_WHISPER_API_KEY", "s3cret")
    assert Config.load(env_path=env_file).openai_api_key == "s3cret"


def test_local_model_override(monkeypatch, env_file, health):
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "local")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "whisper-large-v3-turbo")
    assert Config.load(env_path=env_file).model == "whisper-large-v3-turbo"


def test_explicit_groq_without_key_raises(monkeypatch, env_file, health):
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "groq")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        Config.load(env_path=env_file)


def test_explicit_openai_ignores_groq_key(monkeypatch, env_file, health):
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "openai")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load(env_path=env_file)
    assert cfg.provider == "openai"
    assert cfg.openai_api_key == "sk-test"
    assert cfg.base_url is None


def test_unknown_provider_raises(monkeypatch, env_file, health):
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "azure")
    with pytest.raises(RuntimeError, match="TRANSCRIBE_PROVIDER"):
        Config.load(env_path=env_file)


def test_health_url_is_derived_from_base_url():
    assert config_module.health_url("http://h:18921/v1") == "http://h:18921/health"
    assert config_module.health_url("http://h:18921/v1/") == "http://h:18921/health"
    assert config_module.health_url("http://h:18921") == "http://h:18921/health"


def test_real_probe_reports_unreachable_server():
    # Port 9 (discard) on loopback is closed on any sane host.
    err = config_module.local_health_error("http://127.0.0.1:9/v1", timeout=0.5)
    assert err
