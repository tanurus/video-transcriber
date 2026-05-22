from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


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

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    pattern = out_dir / (audio_path.stem + ".part%03d.mp3")

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
        str(pattern),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Collect chunks sorted
    chunks = sorted(out_dir.glob(audio_path.stem + ".part" + "*" + ".mp3"))
    if not chunks:
        # If segmenting failed, fall back to returning the original file
        return [audio_path]
    return chunks
