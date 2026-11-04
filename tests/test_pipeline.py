import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.pipeline as P
from app import options as O
from app.config import Config
from app.whisper_client import ChunkResult

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_effective_model_respects_provider():
    opts = O.resolve(preset="fast")
    assert P.effective_model(Config(openai_api_key="k", provider="local"), opts) == "whisper-large-v3-turbo"
    assert P.effective_model(Config(openai_api_key="k", model="gpt-4o-transcribe", provider="openai"), opts) == "gpt-4o-transcribe"


def test_build_client_sends_decoding_fields_only_to_local():
    opts = O.resolve({"hotwords": "Kayak", "prompt": "Daily sync"})
    local = P.build_client(Config(openai_api_key="k", provider="local", base_url="http://x/v1"), opts)
    assert local.extra_body["hotwords"] == "Kayak" and local.prompt == "Daily sync"
    assert local._use_verbose is True
    groq = P.build_client(Config(openai_api_key="k", provider="groq"), opts)
    assert groq.extra_body is None and groq.prompt == "Daily sync"


def test_languages_are_weighted_by_speech_time():
    r = [ChunkResult("a", [{"start": 0, "end": 30, "text": "a"}], "ru"),
         ChunkResult("b", [{"start": 0, "end": 5, "text": "b"}], "en"),
         ChunkResult("c", [{"start": 0, "end": 10, "text": "c"}], "RU")]
    assert P._languages(r) == ["ru", "en"]


def test_srt_and_timestamps():
    segs = [{"start": 0.0, "end": 2.5, "text": "Hello"}, {"start": 3661.2, "end": 3662.0, "text": "Later"}]
    srt = P.to_srt(segs)
    assert "00:00:00,000 --> 00:00:02,500" in srt and "01:01:01,200" in srt
    assert P.to_timestamped_text(segs).splitlines()[1] == "[01:01:01] Later"


@needs_ffmpeg
def test_transcribe_media_maps_chunk_offsets_and_keeps_audio(tmp_path, monkeypatch):
    # Real ffmpeg: 3 tones separated by 1.5 s silences -> 3 chunks at pauses.
    import subprocess
    src = tmp_path / "meeting.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=f=440:d=6,apad=pad_dur=1.5[a];sine=f=660:d=6,apad=pad_dur=1.5[b];sine=f=880:d=6[c];[a][b][c]concat=n=3:v=0:a=1",
                    "-ar", "16000", "-ac", "1", str(src)], check=True)

    calls = []

    class FakeClient:
        model = "whisper-large-v3"

        def transcribe_detailed(self, path, log):
            calls.append(Path(path).name)
            return ChunkResult(f"text{len(calls)}", [{"start": 1.0, "end": 2.0, "text": f"text{len(calls)}"}], "ro")

    monkeypatch.setattr(P, "build_client", lambda cfg, opts, timeout=None: FakeClient())
    kept = tmp_path / "audio" / "job.flac"
    opts = O.resolve({"chunk_target_sec": 5, "chunk_max_sec": 8, "max_concurrency": 1})
    res = P.transcribe_media(src, Config(openai_api_key="k", provider="local"), opts, kept, logger=print)

    assert kept.exists(), "lossless audio must be kept for regenerate"
    assert res.chunks == 3 and len(calls) == 3
    starts = [s["start"] for s in res.segments]
    assert starts[0] == 1.0 and 7.0 < starts[1] < 9.0 and 14.0 < starts[2] < 17.0
    assert res.languages == ["ro"]
    assert res.duration and 20 < res.duration < 23

    # A regenerate reuses the kept audio even if the upload is gone.
    src.unlink()
    res2 = P.transcribe_media(src, Config(openai_api_key="k", provider="local"),
                              O.resolve({"segmentation": "native"}), kept)
    assert res2.chunks == 1
