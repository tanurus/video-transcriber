"""Free, deterministic checks on a transcribed chunk (no model calls).

These run before any paid validator: most bad chunks are caught here from
Whisper's own signals and the text itself.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_WORD = re.compile(r"\w+", re.UNICODE)


def repeated_ngram(text: str, min_repeats: int = 3, max_n: int = 6) -> Optional[Dict[str, Any]]:
    """The longest phrase of 1..max_n words repeated back-to-back >= min_repeats times.

    "nesu pasak nesu pasak nesu pasak" -> {"phrase": "nesu pasak", "repeats": 3}.
    Single words need one more repeat (min_repeats + 1) so a natural "no, no, no"
    is not flagged.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    best: Optional[Dict[str, Any]] = None
    for n in range(max_n, 0, -1):
        need = min_repeats + (1 if n == 1 else 0)
        i = 0
        while i + n * need <= len(words):
            gram = words[i:i + n]
            reps = 1
            while words[i + reps * n: i + (reps + 1) * n] == gram:
                reps += 1
            if reps >= need and (best is None or reps * n > best["repeats"] * len(best["phrase"].split())):
                best = {"phrase": " ".join(gram), "repeats": reps}
            i += max(1, reps * n if reps >= need else 1)
        if best:
            return best
    return best


def _weighted(segments: Sequence[Dict[str, Any]], key: str, default: float) -> Optional[float]:
    total = acc = 0.0
    for s in segments:
        d = max(0.01, float(s.get("end", 0)) - float(s.get("start", 0)))
        acc += float(s.get(key, default)) * d
        total += d
    return round(acc / total, 4) if total else None


def check_chunk(
    text: str,
    segments: Sequence[Dict[str, Any]],
    language: Optional[str],
    allowed_languages: Iterable[str] = ("ro", "ru", "en"),
    dropped: Sequence[Any] = (),
    compression_ratio_max: float = 2.4,
    no_speech_max: float = 0.6,
    chunk_duration: Optional[float] = None,
    min_coverage: float = 0.6,
) -> Dict[str, Any]:
    """Verdict + the signals behind it. ``passed`` is True only when nothing is wrong.

    ``segments`` are chunk-relative. With ``chunk_duration`` the check also
    measures how much of the chunk the segments cover: Whisper silently drops a
    sentence in a second language, which shows up as a long uncovered gap.
    """
    allowed = [a.strip().lower() for a in allowed_languages if a and a.strip()]
    lang = (language or "").lower() or None
    max_cr = max((float(s.get("compression_ratio", 0) or 0) for s in segments), default=0.0)
    loop = repeated_ngram(text)
    coverage, gap = speech_coverage(segments, chunk_duration)
    signals = {
        "language": lang,
        "language_allowed": (lang in allowed) if lang else None,
        "avg_logprob": _weighted(segments, "avg_logprob", 0.0),
        "no_speech_prob": _weighted(segments, "no_speech_prob", 0.0),
        "max_compression_ratio": round(max_cr, 3),
        "repeated_phrase": loop,
        "dropped_segments": len(dropped),
        "words": len(_WORD.findall(text)),
        "speech_coverage": coverage,
        "largest_gap": gap,
        "words_per_sec": round(len(_WORD.findall(text)) / chunk_duration, 2) if chunk_duration else None,
    }
    problems: List[str] = []
    if lang and allowed and lang not in allowed:
        problems.append(f"wrong_language:{lang}")
    if max_cr > compression_ratio_max:
        problems.append("loop:compression_ratio")
    if loop:
        problems.append("loop:repeated_phrase")
    if not text.strip():
        problems.append("empty" if not dropped else "empty_after_filter")
    elif coverage is not None and chunk_duration and chunk_duration >= 4 and coverage < min_coverage \
            and gap and gap["end"] - gap["start"] >= 2.0:
        problems.append("low_coverage")
    # Borderline: passed the hard checks but Whisper itself was unsure.
    borderline = not problems and (
        (signals["avg_logprob"] is not None and signals["avg_logprob"] < -0.8)
        or (signals["no_speech_prob"] is not None and signals["no_speech_prob"] > no_speech_max / 2)
        or bool(dropped)
    )
    return {"passed": not problems, "problems": problems, "borderline": borderline, "signals": signals}


def speech_coverage(segments: Sequence[Dict[str, Any]], chunk_duration: Optional[float]):
    """(share of the chunk covered by segments, the longest uncovered gap) — chunk-relative."""
    if not chunk_duration or chunk_duration <= 0:
        return None, None
    spans = sorted((max(0.0, float(s.get("start", 0))), min(chunk_duration, float(s.get("end", 0)))) for s in segments)
    covered, cursor, best = 0.0, 0.0, None
    for a, b in spans:
        if a > cursor and (best is None or a - cursor > best[1] - best[0]):
            best = (cursor, a)
        if b > cursor:
            covered += b - max(a, cursor)
            cursor = b
    if chunk_duration > cursor and (best is None or chunk_duration - cursor > best[1] - best[0]):
        best = (cursor, chunk_duration)
    gap = {"start": round(best[0], 2), "end": round(best[1], 2)} if best else None
    return round(min(1.0, covered / chunk_duration), 3), gap
