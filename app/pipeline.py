"""The web app's transcription pipeline: settings-driven, lossless, timestamped.

source media ──► kept lossless FLAC (reused on regenerate)
                    └─► prepared FLAC (cleanup + loudness)
                           └─► chunks at pauses (or the whole file, "native")
                                  └─► Whisper ──► text + segments on the recording's own timeline
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import options as O
from .audio import extract_lossless, prepare_audio, probe_duration, split_by_silence
from .config import Config
from .whisper_client import ChunkResult, SegmentFilter, WhisperClient

Logger = Callable[[str], None]


@dataclass
class TranscriptResult:
    text: str
    segments: List[Dict[str, Any]]
    languages: List[str]
    duration: Optional[float]
    provider: str
    model: str
    chunks: int = 1
    notes: List[str] = field(default_factory=list)


def effective_model(cfg: Config, opts: Dict[str, Any]) -> str:
    # Both the local server and Groq serve large-v3 and large-v3-turbo; OpenAI has
    # neither id, so its configured model stays.
    if cfg.provider in ("local", "groq"):
        return opts.get("model") or cfg.model
    return cfg.model


def build_client(cfg: Config, opts: Dict[str, Any], timeout: Optional[int] = None) -> WhisperClient:
    local = cfg.provider == "local"
    return WhisperClient(
        api_key=cfg.openai_api_key,
        model=effective_model(cfg, opts),
        timeout=timeout or cfg.timeout,
        base_url=cfg.base_url,
        language=O.fixed_language(opts),
        candidate_languages=O.candidate_list(opts),
        verbose_capable=True if local else None,
        prompt=opts.get("prompt") or None,
        extra_body=O.decode_extra(opts) if local else None,
        segment_filter=SegmentFilter(
            no_speech_threshold=opts["no_speech_threshold"],
            logprob_threshold=opts["logprob_threshold"],
            compression_ratio_threshold=opts["compression_ratio_threshold"],
        ),
    )


def _languages(results: List[ChunkResult]) -> List[str]:
    """Detected languages, weighted by speech duration, most spoken first."""
    tally: Counter = Counter()
    for r in results:
        if r.language:
            spoken = sum(max(0.0, s["end"] - s["start"]) for s in r.segments) or 1.0
            tally[r.language.lower()] += spoken
    return [lang for lang, _ in tally.most_common()]


def transcribe_media(
    source: Path,
    cfg: Config,
    opts: Dict[str, Any],
    kept_audio: Path,
    logger: Optional[Logger] = None,
    rnnoise_model: Optional[str] = None,
) -> TranscriptResult:
    """Transcribe ``source`` (or the already-kept ``kept_audio``) with ``opts``.

    ``kept_audio`` is created from ``source`` when missing and left in place —
    it is what makes regeneration possible after the upload is purged.
    """
    log = logger or (lambda _m: None)
    work = Path(tempfile.mkdtemp(prefix="avtx_job_"))
    try:
        if kept_audio.exists():
            log("Using the kept lossless audio (no re-extraction).")
        else:
            log("Extracting lossless audio (16 kHz mono FLAC)...")
            extract_lossless(source, kept_audio)
        duration = probe_duration(kept_audio)
        if duration:
            log(f"Duration: {duration / 60:.1f} min")

        log("Preparing audio...")
        prepared = prepare_audio(
            kept_audio, work / "prepared.flac",
            normalize=opts["normalize"], denoise=opts["denoise"], highpass=opts["highpass"],
            rnnoise_model=rnnoise_model, log=log,
        )

        native = opts["segmentation"] == "native" and cfg.provider == "local"
        if opts["segmentation"] == "native" and not native:
            log("Whisper native mode needs the local GPU server; using smart chunks instead.")
        if native:
            parts = [(prepared, 0.0)]
            log("Whisper native: sending the whole recording with Whisper's own speech detection.")
        else:
            parts = split_by_silence(
                prepared, opts["chunk_target_sec"], opts["chunk_max_sec"],
                noise_db=f"{opts['silence_noise_db']}dB", min_silence_sec=opts["silence_min_sec"],
                out_dir=work / "chunks",
            )
            log(f"Split into {len(parts)} segment(s) at pauses "
                f"({opts['chunk_target_sec']}-{opts['chunk_max_sec']} s).")

        # A whole-recording decode can run far past the per-chunk timeout.
        timeout = int(max(cfg.timeout, (duration or 0) * 1.5 + 300)) if native else None
        client = build_client(cfg, opts, timeout=timeout)
        where = f" at {cfg.base_url}" if cfg.provider == "local" else ""
        log(f"Backend: {cfg.provider}, model {client.model}{where}")
        log(f"Settings: {O.summary(opts)}")

        results: List[Optional[ChunkResult]] = [None] * len(parts)
        workers = 1 if native else max(1, min(opts["max_concurrency"], len(parts)))
        log(f"Transcribing {len(parts)} segment(s), up to {workers} in parallel...")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(client.transcribe_detailed, p, log): i for i, (p, _o) in enumerate(parts)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()  # a failed segment fails the job
                done += 1
                log(f"Segment {done}/{len(parts)} done.")

        segments: List[Dict[str, Any]] = []
        texts: List[str] = []
        for (path, offset), res in zip(parts, results):
            assert res is not None
            if res.text:
                texts.append(res.text.strip())
            for seg in res.segments:
                seg = dict(seg)
                seg["start"] = round(seg["start"] + offset, 2)
                seg["end"] = round(seg["end"] + offset, 2)
                if res.language:
                    seg["language"] = res.language
                segments.append(seg)
        chunk_results = [r for r in results if r is not None]
        return TranscriptResult(
            text="\n\n".join(t for t in texts if t).strip(),
            segments=segments,
            languages=_languages(chunk_results),
            duration=duration,
            provider=cfg.provider,
            model=client.model,
            chunks=len(parts),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def format_timestamp(seconds: float, srt: bool = False) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines += [str(i), f"{format_timestamp(seg['start'], True)} --> {format_timestamp(seg['end'], True)}",
                  seg["text"], ""]
    return "\n".join(lines)


def to_timestamped_text(segments: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{format_timestamp(s['start'])[:8]}] {s['text']}" for s in segments)
