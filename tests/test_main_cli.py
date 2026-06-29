import argparse
from types import SimpleNamespace

import pytest

import app.main as m
from app.config import Config


def test_parse_args_video_defaults_to_none():
    args = m.parse_args([])
    assert args.video is None, "no hardcoded personal filename as the CLI default"


def test_run_cli_requires_video_path(capsys):
    args = argparse.Namespace(video=None, gui=False)
    assert m.run_cli(args) == 2
    err = capsys.readouterr().err
    assert "video" in err.lower()


def test_transcribe_passes_config_language_to_client(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"y" * 10)
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def transcribe_file(self, f):
            return "txt"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)
    cfg = Config(openai_api_key="x", language="fr")

    m.transcribe_video(video, cfg, logger=lambda s: None, out_path=tmp_path / "out.txt")

    assert seen.get("language") == "fr"


def test_transcript_written_via_atomic_replace(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"y" * 10)
    monkeypatch.setattr(m, "extract_audio", lambda video_path, out_dir, bitrate: audio)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def transcribe_file(self, f):
            return "txt"

    monkeypatch.setattr(m, "WhisperClient", FakeClient)

    def boom(src, dst):
        raise OSError("replace blocked")

    # If the final transcript appears even though os.replace failed, the write
    # bypassed the temp-file + atomic-rename path.
    monkeypatch.setattr(m, "os", SimpleNamespace(replace=boom), raising=False)

    out = tmp_path / "out.txt"
    with pytest.raises(OSError, match="replace blocked"):
        m.transcribe_video(video, Config(openai_api_key="x"), logger=lambda s: None, out_path=out)

    assert not out.exists()
