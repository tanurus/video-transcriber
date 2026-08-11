import subprocess
from pathlib import Path

import pytest

import app.audio as audio


@pytest.fixture(autouse=True)
def fake_ffmpeg(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: "ffmpeg")


def _capture_run(monkeypatch, side_effect=None):
    """Replace subprocess.run, recording (cmd, kwargs); optionally raise or create files."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if side_effect is not None:
            side_effect(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    return calls


def _write_segments(count):
    """Side effect that simulates ffmpeg writing segment files for the output pattern."""

    def effect(cmd):
        pattern = Path(cmd[-1])
        for i in range(count):
            pattern.with_name(pattern.name % i).write_bytes(b"seg")

    return effect


def test_chunk_uses_stream_copy_and_fixed_names(tmp_path, monkeypatch):
    src = tmp_path / "talk.mp3"
    src.write_bytes(b"x" * 10)
    calls = _capture_run(monkeypatch, side_effect=_write_segments(2))

    chunks = audio.chunk_audio_by_size(src, target_mb=24, bitrate="96k")

    cmd = calls[0][0]
    assert "copy" in cmd and cmd[cmd.index("copy") - 1] == "-c"
    # Output pattern must not embed the source stem (glob/printf metachars break it)
    assert Path(cmd[-1]).name == "part%03d.mp3"
    assert [c.name for c in chunks] == ["part000.mp3", "part001.mp3"]


def test_chunk_handles_special_chars_in_stem(tmp_path, monkeypatch):
    src = tmp_path / "weird [clip] 100%.mp3"
    src.write_bytes(b"x" * 10)
    _capture_run(monkeypatch, side_effect=_write_segments(2))

    chunks = audio.chunk_audio_by_size(src, target_mb=24, bitrate="96k")

    assert len(chunks) == 2
    assert all(c != src for c in chunks)


def test_chunk_cleans_temp_dir_on_failure(tmp_path, monkeypatch):
    src = tmp_path / "talk.mp3"
    src.write_bytes(b"x")
    created = {}
    real_mkdtemp = audio.tempfile.mkdtemp

    def fake_mkdtemp(prefix="tmp"):
        d = real_mkdtemp(prefix=prefix, dir=tmp_path)
        created["dir"] = d
        return d

    monkeypatch.setattr(audio.tempfile, "mkdtemp", fake_mkdtemp)

    def explode(cmd):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"segment boom")

    _capture_run(monkeypatch, side_effect=explode)

    with pytest.raises(RuntimeError, match="segment boom"):
        audio.chunk_audio_by_size(src, target_mb=24, bitrate="96k")

    assert not Path(created["dir"]).exists()


def test_chunk_cleans_temp_dir_when_no_chunks(tmp_path, monkeypatch):
    src = tmp_path / "talk.mp3"
    src.write_bytes(b"x")
    created = {}
    real_mkdtemp = audio.tempfile.mkdtemp

    def fake_mkdtemp(prefix="tmp"):
        d = real_mkdtemp(prefix=prefix, dir=tmp_path)
        created["dir"] = d
        return d

    monkeypatch.setattr(audio.tempfile, "mkdtemp", fake_mkdtemp)
    _capture_run(monkeypatch)  # ffmpeg "succeeds" but writes nothing

    chunks = audio.chunk_audio_by_size(src, target_mb=24, bitrate="96k")

    assert chunks == [src]
    assert not Path(created["dir"]).exists()


def test_extract_audio_passes_timeout_and_stdin(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    def create_out(cmd):
        Path(cmd[-1]).write_bytes(b"mp3")

    calls = _capture_run(monkeypatch, side_effect=create_out)

    audio.extract_audio(video, out_dir=tmp_path)

    kwargs = calls[0][1]
    assert kwargs.get("timeout"), "ffmpeg must run with a timeout so a hang cannot wedge the worker"
    assert kwargs.get("stdin") is subprocess.DEVNULL


def test_extract_audio_surfaces_ffmpeg_stderr(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    def explode(cmd):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"Unknown decoder 'xyz'")

    _capture_run(monkeypatch, side_effect=explode)

    with pytest.raises(RuntimeError, match="Unknown decoder"):
        audio.extract_audio(video, out_dir=tmp_path)


# --- silence-aware chunking (parsing + cut-point math, pure functions) --------

def test_parse_silences_extracts_duration_and_intervals():
    stderr = (
        "  Duration: 00:03:20.50, start: 0.000000, bitrate: 96 kb/s\n"
        "[silencedetect @ 0x1] silence_start: 44.2\n"
        "[silencedetect @ 0x1] silence_end: 45.1 | silence_duration: 0.9\n"
        "[silencedetect @ 0x1] silence_start: 89.0\n"
        "[silencedetect @ 0x1] silence_end: 90.0 | silence_duration: 1.0\n"
    )
    duration, silences = audio._parse_silences(stderr)
    assert duration == pytest.approx(200.5)
    assert silences == [(44.2, 45.1), (89.0, 90.0)]


def test_parse_silences_drops_unclosed_trailing_start():
    # Audio ends mid-silence: a silence_start with no matching silence_end yields
    # no usable interior cut point, so it must be discarded.
    stderr = "Duration: 00:00:50.00, start: 0.0\nsilence_start: 30.0\n"
    duration, silences = audio._parse_silences(stderr)
    assert duration == pytest.approx(50.0)
    assert silences == []


def test_parse_silences_without_duration_returns_none():
    duration, silences = audio._parse_silences("no banner here\n")
    assert duration is None
    assert silences == []


def test_compute_cut_points_short_audio_is_single_chunk():
    assert audio._compute_cut_points(60.0, [], target_sec=45, max_sec=90) == []


def test_compute_cut_points_snaps_to_pause_past_target():
    # 200s audio; pauses sit at midpoints 50.5 and 95.5. The first is >= target
    # (45) and <= max (90), so the first cut snaps to that pause rather than a
    # hard time cut.
    cuts = audio._compute_cut_points(
        200.0, [(50.0, 51.0), (95.0, 96.0)], target_sec=45, max_sec=90
    )
    assert cuts[0] == pytest.approx(50.5)
    assert cuts[1] == pytest.approx(95.5)


def test_compute_cut_points_forces_max_when_no_pause():
    # No pauses at all -> hard cuts every max_sec so chunks stay bounded.
    cuts = audio._compute_cut_points(300.0, [], target_sec=45, max_sec=90)
    assert cuts == [90.0, 180.0, 270.0]


def test_compute_cut_points_ignores_pauses_before_target():
    # A very early pause (before target_sec accumulates) must not cause a tiny
    # first chunk; the cut should still land near max.
    cuts = audio._compute_cut_points(120.0, [(5.0, 6.0)], target_sec=45, max_sec=90)
    assert cuts[0] == pytest.approx(90.0)
