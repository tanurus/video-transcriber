from pathlib import Path

from web.settings import WebSettings


def test_load_defaults(monkeypatch):
    for var in ["DATA_DIR", "RETAIN_VIDEO_DAYS", "MAX_CONTENT_MB", "ALLOWED_EXT"]:
        monkeypatch.delenv(var, raising=False)
    s = WebSettings.load()
    assert s.data_dir == Path("/data")
    assert s.retain_video_days == 30
    assert s.max_content_mb == 2048
    assert "mp4" in s.allowed_ext
    assert s.uploads_dir == Path("/data/uploads")
    assert s.transcripts_dir == Path("/data/transcripts")
    assert s.db_path == Path("/data/app.db")


def test_load_overrides(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/tmp/d")
    monkeypatch.setenv("RETAIN_VIDEO_DAYS", "7")
    monkeypatch.setenv("MAX_CONTENT_MB", "100")
    monkeypatch.setenv("ALLOWED_EXT", "mp4, .MKV ,mov")
    s = WebSettings.load()
    assert s.data_dir == Path("/tmp/d")
    assert s.retain_video_days == 7
    assert s.max_content_mb == 100
    assert s.allowed_ext == frozenset({"mp4", "mkv", "mov"})
