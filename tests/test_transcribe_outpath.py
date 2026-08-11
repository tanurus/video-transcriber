from pathlib import Path

import app.main as m
from app.config import Config


def test_transcribe_writes_to_out_path(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"y" * 10)  # small => no chunking

    # Fake the two external boundaries: ffmpeg extraction and the Groq client.
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f, logger=None):
            return "hello world"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)

    cfg = Config(openai_api_key="x")
    out = tmp_path / "out" / "result.txt"

    result = m.transcribe_video(video, cfg, logger=lambda s: None, out_path=out)

    assert result == out
    assert out.read_text(encoding="utf-8") == "hello world"


def test_transcribe_defaults_next_to_video(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"y" * 10)
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f, logger=None):
            return "abc"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)
    cfg = Config(openai_api_key="x")

    result = m.transcribe_video(video, cfg, logger=lambda s: None)

    assert result == video.with_suffix(".txt")
    assert result.read_text(encoding="utf-8") == "abc"
