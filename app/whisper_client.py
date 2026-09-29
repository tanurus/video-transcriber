from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


@dataclass
class ChunkResult:
    """One transcribed audio file: text plus timestamped segments (chunk-relative)."""

    text: str
    segments: List[Dict[str, Any]]
    language: Optional[str] = None
    duration: Optional[float] = None


def filter_segments(
    segments: Iterable[Any], flt: SegmentFilter
) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (kept_text, dropped) from verbose_json segments.

    ``dropped`` is a list of (segment_text, reason) for logging so filtering is
    never silent. Pure function — no I/O — so it is easy to unit test.
    """
    text, _kept, dropped = split_segments(segments, flt)
    return text, dropped


def _plain_segment(seg: Any) -> Dict[str, Any]:
    return {
        "start": float(_seg_get(seg, "start", 0.0)),
        "end": float(_seg_get(seg, "end", 0.0)),
        "text": str(_seg_get(seg, "text", "")).strip(),
        "avg_logprob": float(_seg_get(seg, "avg_logprob", 0.0)),
        "no_speech_prob": float(_seg_get(seg, "no_speech_prob", 0.0)),
    }


def split_segments(
    segments: Iterable[Any], flt: SegmentFilter
) -> Tuple[str, List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Like filter_segments, but also returns the kept segments as plain dicts."""
    kept_segs: List[Dict[str, Any]] = []
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
            kept_segs.append(_plain_segment(seg))

    return " ".join(kept).strip(), kept_segs, dropped


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


def _score_segments(segments: Iterable[Any]) -> float:
    """Duration-weighted mean of avg_logprob — how well a decode fits the audio.

    Used to pick the winning language in a candidate race: the correct language
    fits the acoustics best and scores highest; wrong languages (e.g. Polish for
    Russian speech) decode with markedly lower confidence.
    """
    total_dur = 0.0
    acc = 0.0
    for seg in segments:
        start = float(_seg_get(seg, "start", 0.0))
        end = float(_seg_get(seg, "end", 0.0))
        dur = end - start
        if dur <= 0:
            dur = 1.0
        acc += float(_seg_get(seg, "avg_logprob", -10.0)) * dur
        total_dur += dur
    if total_dur <= 0:
        return float("-inf")
    return acc / total_dur


class WhisperClient:
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3",
        timeout: int = 600,
        base_url: str | None = None,
        language: str | None = None,
        segment_filter: SegmentFilter | None = None,
        candidate_languages: List[str] | None = None,
        verbose_capable: bool | None = None,
        prompt: str | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self.model = model
        self.timeout = timeout
        self.language = language
        self.segment_filter = segment_filter
        # When set, each chunk is transcribed once per candidate and the highest-
        # confidence result wins — this pins Whisper to a known language set (e.g.
        # ro/ru/en) so it can never drift into Polish/Ukrainian.
        self.candidate_languages = candidate_languages or None
        # Spelling/vocabulary hint (all providers) and extra decoding fields (only
        # the local GPU server understands them; cloud APIs would reject them).
        self.prompt = prompt or None
        self.extra_body = dict(extra_body) if extra_body else None
        # None = infer from the model name. The local GPU server always returns
        # segment stats whatever its model is called, so it passes True.
        self._verbose_ok = _supports_verbose(model) if verbose_capable is None else verbose_capable
        # Filtering and the race both need per-segment stats -> verbose_json.
        self._use_verbose = (
            segment_filter is not None or bool(self.candidate_languages)
        ) and self._verbose_ok

    def _create_with_retry(self, audio_file: str | Path, response_format: str, language: str | None):
        retries = 3
        backoff = 5
        for attempt in range(1, retries + 1):
            try:
                with open(audio_file, "rb") as f:
                    # temperature=0 minimises Whisper's repetition/hallucination loops.
                    kwargs = dict(
                        model=self.model,
                        file=f,
                        response_format=response_format,
                        temperature=0,
                    )
                    if language:
                        kwargs["language"] = language
                    if self.prompt:
                        kwargs["prompt"] = self.prompt
                    if self.extra_body:
                        kwargs["extra_body"] = self.extra_body
                    return self.client.audio.transcriptions.create(**kwargs)
            except _RETRYABLE_ERRORS:
                if attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                raise
        raise RuntimeError("unreachable: retry loop exited without returning or raising")

    def transcribe_file(
        self, audio_file: str | Path, logger: Callable[[str], None] | None = None
    ) -> str:
        return self.transcribe_detailed(audio_file, logger).text

    def transcribe_detailed(
        self, audio_file: str | Path, logger: Callable[[str], None] | None = None
    ) -> ChunkResult:
        """Transcribe one file; segments are relative to the start of that file."""
        if self.candidate_languages and self._verbose_ok:
            return self._transcribe_race(audio_file, logger)
        response_format = "verbose_json" if self._use_verbose else "text"
        resp = self._create_with_retry(audio_file, response_format, self.language)
        return self._finalize_detailed(resp, logger)

    def _transcribe_race(
        self, audio_file: str | Path, logger: Callable[[str], None] | None
    ) -> ChunkResult:
        """Transcribe once per candidate language; keep the most confident decode."""
        best: Optional[Tuple[float, str, Any, list]] = None
        for lang in self.candidate_languages:  # type: ignore[union-attr]
            resp = self._create_with_retry(audio_file, "verbose_json", lang)
            segments = _seg_get(resp, "segments", None) or []
            scored_segments = segments
            if self.segment_filter is not None:
                # Rejected hallucinations must not outrank usable speech.
                scored_segments = [
                    seg for seg in segments
                    if filter_segments([seg], self.segment_filter)[0].strip()
                ]
            score = _score_segments(scored_segments)
            if best is None or score > best[0]:
                best = (score, lang, resp, segments)

        score, lang, resp, segments = best  # type: ignore[misc]
        if logger:
            logger(f"  Language race: picked '{lang}' (confidence {score:.3f})")
        result = self._finalize_detailed(resp, logger, segments=segments)
        result.language = lang
        return result

    def _finalize(
        self,
        resp: Any,
        logger: Callable[[str], None] | None,
        segments: Any = None,
    ) -> str:
        return self._finalize_detailed(resp, logger, segments=segments).text

    def _finalize_detailed(
        self,
        resp: Any,
        logger: Callable[[str], None] | None,
        segments: Any = None,
    ) -> ChunkResult:
        language = _seg_get(resp, "language", None) if not isinstance(resp, str) else None
        duration = _seg_get(resp, "duration", None) if not isinstance(resp, str) else None
        if self._use_verbose:
            if segments is None:
                segments = _seg_get(resp, "segments", None)
            if segments:
                if self.segment_filter is not None:
                    text, kept, dropped = split_segments(segments, self.segment_filter)
                    if dropped and logger:
                        logger(f"  Filtered {len(dropped)} low-confidence segment(s):")
                        for seg_text, reason in dropped:
                            preview = (seg_text[:60] + "…") if len(seg_text) > 60 else seg_text
                            logger(f"    - [{reason}] {preview!r}")
                else:
                    kept = [_plain_segment(s) for s in segments if str(_seg_get(s, "text", "")).strip()]
                    text = " ".join(s["text"] for s in kept).strip()
                return ChunkResult(text, kept, language, duration)
            # No segments came back (empty/near-silent audio) -> use plain text.
        text = _extract_text(resp).strip()
        segs = [{"start": 0.0, "end": float(duration or 0.0), "text": text}] if text else []
        return ChunkResult(text, segs, language, duration)

    def transcribe_many(self, audio_files: Iterable[str | Path]) -> str:
        parts: List[str] = []
        for idx, p in enumerate(audio_files, start=1):
            text = self.transcribe_file(p)
            header = f"\n\n--- Segment {idx} ---\n\n"
            parts.append(header + text.strip())
        return "".join(parts).strip() or ""
