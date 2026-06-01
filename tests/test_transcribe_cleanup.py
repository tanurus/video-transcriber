from pathlib import Path
import tempfile

import app.main as m
from app.config import Config


def test_transcribe_cleans_temp_work_dir(tmp_path, monkeypatch):
    """The temporary working directory must be removed after transcription."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"y" * 10)  # small => no chunking

    created = {}
    real_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(prefix="tmp"):
        d = real_mkdtemp(prefix=prefix, dir=tmp_path)
        created["work"] = d
        return d

    monkeypatch.setattr(m.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f):
            return "txt"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)

    out = tmp_path / "out.txt"
    m.transcribe_video(video, Config(openai_api_key="x"), logger=lambda s: None, out_path=out)

    assert "work" in created
    assert not Path(created["work"]).exists()  # temp work dir cleaned up
    assert out.read_text(encoding="utf-8") == "txt"


def test_transcribe_cleans_temp_dir_on_failure(tmp_path, monkeypatch):
    """Even if transcription raises, the temp working dir must be removed."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"y" * 10)

    created = {}
    real_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(prefix="tmp"):
        d = real_mkdtemp(prefix=prefix, dir=tmp_path)
        created["work"] = d
        return d

    monkeypatch.setattr(m.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class BoomClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f):
            raise RuntimeError("api down")

    monkeypatch.setattr(m, "WhisperClient", BoomClient)

    import pytest

    with pytest.raises(RuntimeError):
        m.transcribe_video(video, Config(openai_api_key="x"), logger=lambda s: None)

    assert not Path(created["work"]).exists()  # cleaned up despite the error
