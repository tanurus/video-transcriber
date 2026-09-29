from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


FFMPEG_TIMEOUT_SECONDS = 3600  # a hung ffmpeg must never wedge the worker forever

# Under pythonw.exe (the no-console GUI launch) a child console app would other-
# wise get a brand-new console window that flashes up and steals focus. Windows-
# only flag; empty dict elsewhere so the same **kwargs splat is a no-op.
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def _which_ffmpeg() -> Optional[str]:
    # On Windows this should find ffmpeg.exe if on PATH
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/download.html and ensure ffmpeg.exe is on PATH."
        )
    return ffmpeg


def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            **_NO_WINDOW,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace").strip() if e.stderr else ""
        detail = stderr[-500:] or "no stderr output"
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}): {detail}") from e


def extract_audio(
    video_path: str | Path,
    out_dir: Optional[str | Path] = None,
    bitrate: str = "96k",
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract audio from a video file using ffmpeg.

    - Outputs MP3 mono at given bitrate and sample rate for consistent transcription.
    - Returns the path to the generated audio file.
    """
    ffmpeg = require_ffmpeg()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="avtx_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_dir / (video_path.stem + ".mp3")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        str(bitrate),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]

    _run_ffmpeg(cmd)
    return audio_path


_BITRATE_RE = re.compile(r"^(\d+)([kKmM]?)$")


def _bitrate_to_bps(bitrate: str) -> int:
    m = _BITRATE_RE.match(bitrate.strip())
    if not m:
        # fallback to 96k if unparsable
        return 96000
    value = int(m.group(1))
    suffix = m.group(2).lower()
    if suffix == "m":
        return value * 1_000_000
    if suffix == "k":
        return value * 1_000
    return value


def chunk_audio_by_size(
    audio_path: str | Path,
    target_mb: int = 24,
    bitrate: str = "96k",
    min_segment_sec: int = 60,
) -> List[Path]:
    """Chunk an audio file into segments near the target size using ffmpeg segmenting.

    We estimate segment duration from target size and bitrate: size ~= bitrate * duration.
    Ensures a floor on segment duration to avoid overly small chunks.
    Returns a list of chunk file paths (in a temp directory alongside the input).
    """
    ffmpeg = require_ffmpeg()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    bps = _bitrate_to_bps(bitrate)
    # target bytes ~= target_mb * 1024*1024; bytes = bits/8 => duration = bytes*8/bps
    target_seconds = max(int((target_mb * 1024 * 1024 * 8) / max(bps, 1)), min_segment_sec)

    out_dir = Path(tempfile.mkdtemp(prefix="avtx_chunks_"))
    # Fixed segment names: the source stem must not reach ffmpeg's printf
    # pattern ('%' breaks it) or the glob below ('[', '*', '?' break matching).
    pattern = out_dir / "part%03d.mp3"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(target_seconds),
        "-reset_timestamps",
        "1",
        # Stream-copy: re-encoding at ffmpeg's default bitrate would inflate
        # chunks past the target size (and waste a full encode pass).
        "-c",
        "copy",
        str(pattern),
    ]

    try:
        _run_ffmpeg(cmd)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    chunks = sorted(out_dir.glob("part*.mp3"))
    if not chunks:
        # If segmenting produced nothing, fall back to the original file
        shutil.rmtree(out_dir, ignore_errors=True)
        return [audio_path]
    return chunks


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def _parse_silences(stderr: str) -> Tuple[Optional[float], List[Tuple[float, float]]]:
    """Parse ffmpeg silencedetect stderr into (total_duration, [(start, end), ...]).

    Duration comes from ffmpeg's own "Duration: HH:MM:SS.ss" banner (we have no
    ffprobe). A trailing silence_start with no matching silence_end (audio ends
    mid-silence) is dropped since it yields no usable interior cut point.
    """
    duration: Optional[float] = None
    m = _DURATION_RE.search(stderr)
    if m:
        h, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = h * 3600 + mm * 60 + ss

    starts = [float(x) for x in _SILENCE_START_RE.findall(stderr)]
    ends = [float(x) for x in _SILENCE_END_RE.findall(stderr)]
    silences = [(s, e) for s, e in zip(starts, ends) if e > s]
    return duration, silences


def _compute_cut_points(
    duration: float,
    silences: List[Tuple[float, float]],
    target_sec: float,
    max_sec: float,
) -> List[float]:
    """Greedy cut points: after each cut, prefer the next pause past target_sec,
    but force a cut at max_sec if no pause falls in the [target, max] window.

    Returns interior cut timestamps (strictly between 0 and duration). An empty
    list means "don't split" (audio already fits in one chunk).
    """
    if duration <= max_sec:
        return []

    mids = sorted(m for m in ((s + e) / 2 for s, e in silences) if 0 < m < duration)

    cuts: List[float] = []
    last = 0.0
    idx = 0
    # Keep cutting while the tail after the last cut is still longer than max_sec.
    while duration - last > max_sec:
        target_point = last + target_sec
        max_point = last + max_sec
        # Skip pauses that fall before we've accumulated target_sec of audio.
        while idx < len(mids) and mids[idx] < target_point:
            idx += 1
        if idx < len(mids) and mids[idx] <= max_point:
            chosen = mids[idx]
            idx += 1
        else:
            chosen = max_point  # no pause in range -> hard cut so chunks stay bounded
        cuts.append(round(chosen, 3))
        last = chosen
    return cuts


def chunk_audio_by_silence(
    audio_path: str | Path,
    target_sec: int = 45,
    max_sec: int = 90,
    noise_db: str = "-30dB",
    min_silence_sec: float = 0.5,
) -> List[Path]:
    """Chunk audio into ~target..max second segments, snapping cuts to pauses.

    Short chunks make Whisper re-detect the spoken language per segment, which
    is what stops long multilingual recordings from being decoded through a
    single wrong language. Cutting on silence avoids slicing words mid-sentence.

    Raises on hard ffmpeg failure or if the duration can't be read, so callers
    can fall back to size-based chunking.
    """
    ffmpeg = require_ffmpeg()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    detect_cmd = [
        ffmpeg,
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    # silencedetect writes its report to stderr; -f null discards the audio.
    proc = _run_ffmpeg(detect_cmd)
    stderr = proc.stderr.decode("utf-8", "replace") if proc.stderr else ""
    duration, silences = _parse_silences(stderr)
    if duration is None:
        raise RuntimeError("could not determine audio duration from ffmpeg output")

    cut_points = _compute_cut_points(duration, silences, target_sec, max_sec)
    if not cut_points:
        return [audio_path]

    out_dir = Path(tempfile.mkdtemp(prefix="avtx_chunks_"))
    pattern = out_dir / "part%03d.mp3"
    segment_cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_times",
        ",".join(f"{c:.3f}" for c in cut_points),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(pattern),
    ]
    try:
        _run_ffmpeg(segment_cmd)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    chunks = sorted(out_dir.glob("part*.mp3"))
    if not chunks:
        shutil.rmtree(out_dir, ignore_errors=True)
        return [audio_path]
    return chunks


# --- lossless preparation pipeline (web app) -------------------------------------
#
# The original MP3 path above re-encodes lossy audio before Whisper. The web
# pipeline instead keeps one lossless 16 kHz mono FLAC per recording (so a job
# can be regenerated long after the uploaded video is purged) and derives the
# prepared audio from it with an explicit, logged filter chain.

LUFS_TARGET = -16.0
MAX_GAIN_DB = 20.0
_LUFS_RE = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)

DENOISE_FILTERS = {
    "light": "afftdn=nr=10:nf=-40:tn=1",
    "strong": "afftdn=nr=25:nf=-30:tn=1",
}


def probe_duration(path: str | Path) -> Optional[float]:
    """Media duration in seconds (ffprobe, falling back to ffmpeg's banner)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = _run_ffmpeg([ffprobe, "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
            return float(proc.stdout.decode().strip())
        except Exception:  # noqa: BLE001 - fall through to the banner parse
            pass
    try:
        proc = subprocess.run([require_ffmpeg(), "-i", str(path)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=120, **_NO_WINDOW)
        duration, _ = _parse_silences(proc.stderr.decode("utf-8", "replace"))
        return duration
    except Exception:  # noqa: BLE001
        return None


def extract_lossless(src: str | Path, dest: str | Path, sample_rate: int = 16000) -> Path:
    """Decode any audio/video into mono 16 kHz FLAC — Whisper's native format, no lossy step."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".partial" + dest.suffix)
    _run_ffmpeg([require_ffmpeg(), "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
                 "-c:a", "flac", str(tmp)])
    tmp.replace(dest)
    return dest


def measure_loudness(path: str | Path, pre_filter: str = "") -> Optional[float]:
    """Integrated loudness (LUFS, EBU R128) of the audio after ``pre_filter``."""
    chain = (pre_filter + "," if pre_filter else "") + "ebur128"
    proc = _run_ffmpeg([require_ffmpeg(), "-nostats", "-i", str(path), "-af", chain, "-f", "null", "-"])
    found = _LUFS_RE.findall(proc.stderr.decode("utf-8", "replace"))
    if not found:
        return None
    value = float(found[-1])
    return None if value <= -70 else value  # -70 = silence gate, nothing to measure


def cleanup_filters(denoise: str = "off", highpass: bool = False,
                    rnnoise_model: Optional[str] = None) -> Tuple[str, List[str]]:
    """ffmpeg filter chain for the cleanup step, plus notes worth logging."""
    parts: List[str] = []
    notes: List[str] = []
    if highpass:
        parts.append("highpass=f=80")
    if denoise == "rnnoise":
        if rnnoise_model and Path(rnnoise_model).exists():
            # RNNoise models are trained at 48 kHz.
            parts.append(f"aresample=48000,arnndn=m={rnnoise_model},aresample=16000")
        else:
            notes.append("RNNoise model not found; using light denoise instead")
            parts.append(DENOISE_FILTERS["light"])
    elif denoise in DENOISE_FILTERS:
        parts.append(DENOISE_FILTERS[denoise])
    return ",".join(parts), notes


def normalization_gain(lufs: Optional[float]) -> float:
    """Single linear gain towards -16 LUFS, capped so room tone is never blown up."""
    if lufs is None:
        return 0.0
    return round(max(-MAX_GAIN_DB, min(MAX_GAIN_DB, LUFS_TARGET - lufs)), 2)


def prepare_audio(
    src: str | Path,
    dest: str | Path,
    normalize: bool = True,
    denoise: str = "off",
    highpass: bool = False,
    rnnoise_model: Optional[str] = None,
    log=None,
) -> Path:
    """Apply cleanup + loudness normalization to lossless audio; writes FLAC."""
    log = log or (lambda _m: None)
    chain, notes = cleanup_filters(denoise, highpass, rnnoise_model)
    for n in notes:
        log(f"  {n}")
    filters = [chain] if chain else []
    if normalize:
        lufs = measure_loudness(src, chain)
        gain = normalization_gain(lufs)
        log(f"  Loudness {lufs if lufs is not None else 'n/a'} LUFS -> gain {gain:+.1f} dB")
        if gain:
            # alimiter only touches peaks, so a boost never turns into a cut.
            filters.append(f"volume={gain}dB,alimiter=limit=0.89:level=false")
    dest = Path(dest)
    if not filters:
        shutil.copyfile(src, dest)
        return dest
    _run_ffmpeg([require_ffmpeg(), "-y", "-i", str(src), "-af", ",".join(filters),
                 "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dest)])
    return dest


def split_by_silence(
    audio_path: str | Path,
    target_sec: float,
    max_sec: float,
    noise_db: str = "-30dB",
    min_silence_sec: float = 0.5,
    out_dir: Optional[str | Path] = None,
) -> List[Tuple[Path, float]]:
    """Silence-aware split that also reports each chunk's start offset (seconds).

    Offsets let chunk-relative Whisper timestamps map back onto the recording.
    """
    audio_path = Path(audio_path)
    proc = _run_ffmpeg([require_ffmpeg(), "-i", str(audio_path), "-af",
                        f"silencedetect=noise={noise_db}:d={min_silence_sec}", "-f", "null", "-"])
    duration, silences = _parse_silences(proc.stderr.decode("utf-8", "replace"))
    if duration is None:
        raise RuntimeError("could not determine audio duration from ffmpeg output")
    cuts = _compute_cut_points(duration, silences, target_sec, max_sec)
    if not cuts:
        return [(audio_path, 0.0)]
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="avtx_chunks_"))
    out.mkdir(parents=True, exist_ok=True)
    suffix = audio_path.suffix or ".flac"
    _run_ffmpeg([require_ffmpeg(), "-y", "-i", str(audio_path), "-f", "segment",
                 "-segment_times", ",".join(f"{c:.3f}" for c in cuts), "-reset_timestamps", "1",
                 "-c", "copy", str(out / f"part%04d{suffix}")])
    chunks = sorted(out.glob(f"part*{suffix}"))
    offsets = [0.0] + cuts
    if len(chunks) != len(offsets):
        # The muxer can merge a sliver at the end; offsets must still line up.
        offsets = offsets[: len(chunks)]
    return list(zip(chunks, offsets))
