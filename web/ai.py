"""AI finishing: clean the transcript text, then title, describe and tag it.

Works with any OpenAI-compatible chat endpoint. Designed to be safe on a
verbatim record:

* the raw transcript is never modified — the cleaned text is stored beside it;
* cleanup runs in chunks, and each cleaned chunk must pass a guard (length
  ratio + word overlap in both directions) or the original chunk is kept, so a
  model that summarises, translates or invents text cannot slip it through.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

CHUNK_CHARS = 5000
MIN_LENGTH_RATIO = 0.45  # cleaned/original characters; fillers alone rarely remove more
MAX_LENGTH_RATIO = 1.15
MIN_KEPT_WORDS = 0.75  # share of the original's distinct content words still present
MIN_GROUNDED_WORDS = 0.85  # share of the cleaned text's distinct words found in the original

CLEAN_SYSTEM = """You clean raw speech-to-text meeting transcripts. Output ONLY the cleaned transcript text.
Rules:
- Keep every language exactly as spoken. NEVER translate. Mixed Russian / Romanian / Ukrainian / English stays mixed.
- Remove filler words and hesitations (um, uh, er, like, you know, ну, э, эм, типа, как бы, ăă, deci when used as filler), stutters, false starts and immediately repeated words.
- Remove obvious speech-recognition artefacts that nobody said: "Thank you for watching", "Subscribe to the channel", "Субтитры сделал ...", "Продолжение следует", "Să vă mulțumim pentru vizionare", music notes, and the same phrase looping.
- Fix punctuation, capitalisation and sentence boundaries. Start a new paragraph when the topic or speaker clearly changes.
- Keep every substantive statement, number, date, amount, name, brand and decision. Keep the speakers' own words.
- Do NOT summarise, reorder, explain, add headings, add speaker labels or add anything that was not said.
- If a passage is unintelligible, keep it as it is."""

META_SYSTEM = """You write catalogue metadata for a meeting / recording transcript.
Reply with ONE JSON object and nothing else:
{"title": "...", "description": "...", "tags": ["...", "..."], "language": "...", "people": ["..."], "topics": ["..."]}
- title: specific and scannable, at most 80 characters, no date, no quotes, no trailing full stop. Say what it is about, e.g. "Kayak partnership: click growth vs flat bookings".
- description: 2-3 sentences: what was discussed and any decisions, owners or numbers.
- tags: 3-6 short lowercase tags.
- language: the main spoken language as an ISO 639-1 code.
- people: names of people mentioned or speaking (may be empty).
- topics: 2-5 main topics.
Base everything strictly on the transcript. Do not invent facts."""

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class AIConfig:
    base_url: str
    api_key: str
    model: str
    language: str = "English"
    timeout: int = 300
    chunk_chars: int = CHUNK_CHARS


@dataclass
class AIResult:
    clean_text: str
    title: str
    description: str
    tags: List[str]
    language: Optional[str] = None
    people: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    kept_chunks: int = 0
    rejected_chunks: int = 0


def split_text(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Split at paragraph, then sentence boundaries, into pieces under ``limit``."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: List[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        units = [para] if len(para) <= limit else re.split(r"(?<=[.!?…])\s+", para)
        for unit in units:
            while len(unit) > limit:  # a monster sentence: hard cut at a space
                cut = unit.rfind(" ", 0, limit) or limit
                pieces.append(unit[:cut].strip())
                unit = unit[cut:]
            if len(buf) + len(unit) + 2 > limit and buf:
                pieces.append(buf.strip())
                buf = ""
            buf += (unit if not buf else ("\n\n" if unit is para else " ") + unit)
    if buf.strip():
        pieces.append(buf.strip())
    return [p for p in pieces if p]


def _content_words(text: str) -> set:
    """Distinct words of 3+ letters: fillers (э, ну, um, uh) and repetition loops
    do not count, so removing them never looks like lost content."""
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= 3}


def guard(original: str, cleaned: str) -> Optional[str]:
    """None if ``cleaned`` is a faithful cleanup of ``original``, else the reason it is not."""
    if not cleaned.strip():
        return "empty output"
    ratio = len(cleaned) / max(1, len(original))
    if ratio > MAX_LENGTH_RATIO:
        return f"too long ({ratio:.0%} of original)"
    ow, cw = _content_words(original), _content_words(cleaned)
    grounded = len(cw & ow) / max(1, len(cw))
    if grounded < MIN_GROUNDED_WORDS:
        return f"introduced new words ({grounded:.0%} grounded) — translated or invented"
    if ratio < MIN_LENGTH_RATIO:
        return f"too short ({ratio:.0%} of original)"
    kept = len(ow & cw) / max(1, len(ow))
    if kept < MIN_KEPT_WORDS:
        return f"dropped too many words ({kept:.0%} kept)"
    return None


def parse_json_object(text: str) -> Dict[str, Any]:
    """The first {...} object in a model reply (tolerates code fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("unterminated JSON object in reply")


def _clip(value: Any, limit: int, strip_period: bool = True) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip('"')
    if strip_period:
        text = text.rstrip(".")
    return text[:limit].strip()


class AIFinisher:
    def __init__(self, cfg: AIConfig, client: Any = None) -> None:
        self.cfg = cfg
        self.client = client or OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout,
                                       max_retries=3)

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    def clean_chunk(self, chunk: str) -> tuple:
        out = self._chat(CLEAN_SYSTEM, chunk)
        # Some models wrap the answer in a code fence despite instructions.
        out = re.sub(r"^```\w*\n|\n```$", "", out).strip()
        reason = guard(chunk, out)
        return (chunk, reason) if reason else (out, None)

    def metadata(self, text: str) -> Dict[str, Any]:
        sample = text if len(text) <= 60000 else text[:45000] + "\n…\n" + text[-15000:]
        lang = self.cfg.language
        lang_rule = ("Write title, description and tags in the transcript's main language."
                     if lang.lower() in ("same", "auto", "") else f"Write title, description and tags in {lang}.")
        user = f"{lang_rule}\n\nTranscript:\n{sample}"
        last_error: Optional[Exception] = None
        for _ in range(2):  # one retry on an unparseable reply
            try:
                return parse_json_object(self._chat(META_SYSTEM, user))
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
        raise RuntimeError(f"metadata reply was not valid JSON: {last_error}")

    def finish(self, text: str, log: Callable[[str], None] = lambda _m: None) -> AIResult:
        chunks = split_text(text, self.cfg.chunk_chars)
        log(f"AI cleanup: {len(chunks)} chunk(s) with {self.cfg.model}")
        with ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(self.clean_chunk, chunks))
        notes = []
        rejected = 0
        for i, (_out, reason) in enumerate(results, start=1):
            if reason:
                rejected += 1
                notes.append(f"chunk {i}: kept original ({reason})")
        clean = "\n\n".join(out for out, _r in results).strip()
        for n in notes:
            log("  " + n)
        meta = self.metadata(clean or text)
        tags = [_clip(t, 40).lower() for t in (meta.get("tags") or []) if _clip(t, 40)][:6]
        return AIResult(
            clean_text=clean,
            title=_clip(meta.get("title"), 80) or "Untitled recording",
            description=_clip(meta.get("description"), 600, strip_period=False),
            tags=tags,
            language=_clip(meta.get("language"), 8).lower() or None,
            people=[_clip(p, 60) for p in (meta.get("people") or []) if _clip(p, 60)][:12],
            topics=[_clip(t, 60) for t in (meta.get("topics") or []) if _clip(t, 60)][:6],
            notes=notes,
            kept_chunks=len(chunks) - rejected,
            rejected_chunks=rejected,
        )

    def list_models(self) -> List[str]:
        return sorted(m.id for m in self.client.models.list())
