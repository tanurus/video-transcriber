from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Only transient failures are worth retrying; 4xx errors (bad key, bad model,
# file too large) would just re-upload the audio three times to fail the same way.
_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


@dataclass
class SegmentFilter:
    """Thresholds for discarding hallucinated / no-speech segments.

    Defaults are Whisper's own internal values. A segment is dropped when it is
    almost certainly silence (high no_speech_prob AND low confidence) or when it
    is a repetition loop (compression_ratio too high, e.g. "rast rast rast...").
    """

    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


def _seg_get(seg: Any, key: str, default: Any) -> Any:
    if isinstance(seg, dict):
        value = seg.get(key, default)
    else:
        value = getattr(seg, key, default)
    return default if value is None else value


def filter_segments(
    segments: Iterable[Any], flt: SegmentFilter
) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (kept_text, dropped) from verbose_json segments.

    ``dropped`` is a list of (segment_text, reason) for logging so filtering is
    never silent. Pure function — no I/O — so it is easy to unit test.
    """
    kept: List[str] = []
    dropped: List[Tuple[str, str]] = []
    for seg in segments:
        text = str(_seg_get(seg, "text", "")).strip()
        no_speech = float(_seg_get(seg, "no_speech_prob", 0.0))
        avg_logprob = float(_seg_get(seg, "avg_logprob", 0.0))
        compression = float(_seg_get(seg, "compression_ratio", 0.0))

        reason: Optional[str] = None
        if compression > flt.compression_ratio_threshold:
            reason = f"repetition (compression_ratio={compression:.2f})"
        elif no_speech > flt.no_speech_threshold and avg_logprob < flt.logprob_threshold:
            reason = f"no speech (no_speech_prob={no_speech:.2f}, avg_logprob={avg_logprob:.2f})"

        if reason:
            dropped.append((text, reason))
        elif text:
            kept.append(text)

    return " ".join(kept).strip(), dropped


def _extract_text(resp: Any) -> str:
    # For response_format="text", resp is a string-like object with .text;
    # some SDK versions return a plain string instead.
    if hasattr(resp, "text") and isinstance(resp.text, str):
        return resp.text
    if isinstance(resp, str):
        return resp
    return str(resp)


def _supports_verbose(model: str) -> bool:
    # whisper-* (Groq whisper-large-v3, OpenAI whisper-1) return verbose_json with
    # per-segment stats; gpt-4o-transcribe does not, so it stays on plain text.
    return "whisper" in model.lower()


class WhisperClient:
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3",
        timeout: int = 600,
        base_url: str | None = None,
        language: str | None = None,
        segment_filter: SegmentFilter | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self.model = model
        self.timeout = timeout
        self.language = language
        # Filtering needs per-segment stats, which only the verbose_json response
        # (whisper models) provides. Disabled -> plain text, no filtering.
        self.segment_filter = segment_filter
        self._use_verbose = segment_filter is not None and _supports_verbose(model)

    def transcribe_file(
        self, audio_file: str | Path, logger: Callable[[str], None] | None = None
    ) -> str:
        retries = 3
        backoff = 5
        response_format = "verbose_json" if self._use_verbose else "text"
        for attempt in range(1, retries + 1):
            try:
                with open(audio_file, "rb") as f:
                    # Transcribe in the SPOKEN language: Whisper auto-detects when
                    # no language is passed, and native-language transcription is
                    # more accurate than translating to English on the fly.
                    # temperature=0 minimises Whisper's repetition/hallucination loops.
                    kwargs = dict(
                        model=self.model,
                        file=f,
                        response_format=response_format,
                        temperature=0,
                    )
                    if self.language:
                        kwargs["language"] = self.language
                    resp = self.client.audio.transcriptions.create(**kwargs)
                return self._parse_response(resp, logger)
            except _RETRYABLE_ERRORS:
                if attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                raise
        raise RuntimeError("unreachable: retry loop exited without returning or raising")

    def _parse_response(self, resp: Any, logger: Callable[[str], None] | None) -> str:
        if self._use_verbose and self.segment_filter is not None:
            segments = _seg_get(resp, "segments", None)
            if segments:
                text, dropped = filter_segments(segments, self.segment_filter)
                if dropped and logger:
                    logger(f"  Filtered {len(dropped)} low-confidence segment(s):")
                    for seg_text, reason in dropped:
                        preview = (seg_text[:60] + "…") if len(seg_text) > 60 else seg_text
                        logger(f"    - [{reason}] {preview!r}")
                return text
            # No segments came back (empty/near-silent audio) -> use plain text.
        return _extract_text(resp)

    def transcribe_many(self, audio_files: Iterable[str | Path]) -> str:
        parts: List[str] = []
        for idx, p in enumerate(audio_files, start=1):
            text = self.transcribe_file(p)
            header = f"\n\n--- Segment {idx} ---\n\n"
            parts.append(header + text.strip())
        return "".join(parts).strip() or ""
