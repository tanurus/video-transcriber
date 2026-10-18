"""Chunk-level transcription sessions — the engine behind the n8n validated pipeline.

A session prepares one recording once (kept lossless audio → normalised →
pause-aligned chunk plan) and then lets a caller transcribe, re-cut or split
any single chunk with its own settings, keeping every attempt. ``assemble``
turns the chosen attempts into a library transcript.

Chunk ids are stable integers: splitting a chunk adds a new id instead of
renumbering, so a workflow iterating over chunks never loses its place.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from app import options as O
from app.audio import (
    extract_lossless,
    extract_segment,
    prepare_audio,
    probe_duration,
    silence_cut_points,
)
from app.quality import check_chunk

from .storage import Storage, utcnow_iso

DEFAULT_ALLOWED = ("ro", "ru", "en")
PARAGRAPH_GAP_SEC = 2.0


class SessionError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        params_json TEXT NOT NULL,
        prepared_path TEXT NOT NULL,
        duration_sec REAL,
        result_job_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_chunks (
        session_id TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        position REAL NOT NULL,
        start_sec REAL NOT NULL,
        end_sec REAL NOT NULL,
        PRIMARY KEY (session_id, chunk_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_attempts (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        chunk_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        start_sec REAL NOT NULL,
        end_sec REAL NOT NULL,
        params_json TEXT NOT NULL,
        text TEXT NOT NULL,
        segments_json TEXT NOT NULL,
        language TEXT,
        model TEXT,
        checks_json TEXT NOT NULL,
        passed INTEGER NOT NULL,
        external_json TEXT,
        duration_ms INTEGER,
        label TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS attempts_chunk ON session_attempts(session_id, chunk_id)",
]


def _params_to_raw(params: Dict[str, Any]) -> Dict[str, Any]:
    """Accept friendly API params and map them onto the settings catalogue.

    ``language: "ro"`` fixes the language; ``candidate_languages: ["ro","ru"]`` (or
    "ro,ru") races them; ``language: "auto"`` restores auto-detect.
    """
    raw = {k: v for k, v in params.items() if k in O.BY_KEY}
    lang = params.get("language")
    cands = params.get("candidate_languages")
    if isinstance(cands, (list, tuple)):
        cands = ",".join(str(c) for c in cands)
    if cands:
        raw["language_mode"] = "candidates"
        raw["candidate_languages"] = cands
    elif lang and str(lang).lower() != "auto":
        raw["language_mode"] = "fixed"
        raw["language"] = str(lang).lower()
    elif lang:
        raw["language_mode"] = "auto"
    return raw


class SessionService:
    def __init__(
        self,
        storage: Storage,
        jobs: Any,
        cfg_loader: Callable[[], Any],
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.storage = storage
        self.jobs = jobs
        self.cfg_loader = cfg_loader
        if client_factory is None:
            from app.pipeline import build_client
            client_factory = build_client
        self.client_factory = client_factory
        with self._connect() as conn:
            for stmt in SCHEMA:
                conn.execute(stmt)

    # --- plumbing -------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return self.storage._connect()

    @property
    def root(self) -> Path:
        d = self.jobs.data_dir / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _session_row(self, sid: str) -> sqlite3.Row:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row is None:
            raise SessionError("Unknown session.", 404)
        return row

    def _chunks(self, sid: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM session_chunks WHERE session_id = ? ORDER BY position", (sid,)
            ).fetchall()

    def _chunk(self, sid: str, chunk_id: int) -> sqlite3.Row:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_chunks WHERE session_id = ? AND chunk_id = ?", (sid, chunk_id)
            ).fetchone()
        if row is None:
            raise SessionError(f"Chunk {chunk_id} is not in this session.", 404)
        return row

    @staticmethod
    def _attempt_json(row: sqlite3.Row, current: Optional[sqlite3.Row] = None) -> Dict[str, Any]:
        stale = current is not None and (
            abs(row["start_sec"] - current["start_sec"]) > 0.01 or abs(row["end_sec"] - current["end_sec"]) > 0.01
        )
        return {
            "attempt_id": row["id"],
            "chunk_id": row["chunk_id"],
            "label": row["label"],
            "created_at": row["created_at"],
            "start": row["start_sec"],
            "end": row["end_sec"],
            "params": json.loads(row["params_json"]),
            "text": row["text"],
            "language": row["language"],
            "model": row["model"],
            "checks": json.loads(row["checks_json"]),
            "passed": bool(row["passed"]),
            "external": json.loads(row["external_json"]) if row["external_json"] else None,
            "duration_ms": row["duration_ms"],
            "stale": stale,
        }

    # --- sessions -------------------------------------------------------------------------

    def create(self, job_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        job = self.storage.get_job(job_id)
        if job is None:
            raise SessionError("Unknown job.", 404)
        root_id = self.storage.root_of(job_id)
        kept = Path(job.audio_path) if job.audio_path else self.jobs.kept_audio_path(root_id)
        if not kept.exists():
            source = self.jobs.upload_file(job_id) or self.jobs.upload_file(root_id)
            if source is None:
                raise SessionError("Neither the kept audio nor the original upload exists any more.", 409)
            extract_lossless(source, kept)
            self.storage.update_job(job_id, audio_path=str(kept))
        base = O.resolve(_params_to_raw(params), base=job.settings or None)
        sid = uuid.uuid4().hex
        sdir = self.root / sid
        sdir.mkdir(parents=True)
        prepared = prepare_audio(kept, sdir / "prepared.flac", normalize=base["normalize"],
                                 denoise=base["denoise"], highpass=base["highpass"],
                                 rnnoise_model=getattr(self.jobs, "rnnoise_model", None))
        duration, cuts = silence_cut_points(
            prepared, base["chunk_target_sec"], base["chunk_max_sec"],
            noise_db=f"{base['silence_noise_db']}dB", min_silence_sec=base["silence_min_sec"],
        )
        bounds = list(zip([0.0] + cuts, cuts + [duration]))
        allowed = params.get("allowed_languages") or list(DEFAULT_ALLOWED)
        if isinstance(allowed, str):
            allowed = [a.strip() for a in allowed.split(",") if a.strip()]
        stored = {"base": base, "allowed_languages": allowed}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, job_id, created_at, status, params_json, prepared_path, duration_sec) "
                "VALUES (?, ?, ?, 'open', ?, ?, ?)",
                (sid, job_id, utcnow_iso(), json.dumps(stored), str(prepared), duration),
            )
            conn.executemany(
                "INSERT INTO session_chunks (session_id, chunk_id, position, start_sec, end_sec) VALUES (?, ?, ?, ?, ?)",
                [(sid, i, float(i), round(s, 3), round(e, 3)) for i, (s, e) in enumerate(bounds)],
            )
        return self.get(sid)

    def get(self, sid: str, include_attempts: bool = False) -> Dict[str, Any]:
        row = self._session_row(sid)
        stored = json.loads(row["params_json"])
        chunks = self._chunks(sid)
        with self._connect() as conn:
            attempts = conn.execute(
                "SELECT * FROM session_attempts WHERE session_id = ? ORDER BY created_at", (sid,)
            ).fetchall()
        by_chunk: Dict[int, List[sqlite3.Row]] = {}
        for a in attempts:
            by_chunk.setdefault(a["chunk_id"], []).append(a)
        out_chunks = []
        for c in chunks:
            mine = [self._attempt_json(a, c) for a in by_chunk.get(c["chunk_id"], [])]
            fresh = [a for a in mine if not a["stale"]]
            best = self._best(fresh)
            item = {
                "chunk_id": c["chunk_id"], "start": c["start_sec"], "end": c["end_sec"],
                "duration": round(c["end_sec"] - c["start_sec"], 3),
                "attempts": len(mine), "passed": any(a["passed"] for a in fresh),
                "best_attempt_id": best["attempt_id"] if best else None,
            }
            if include_attempts:
                item["attempt_list"] = mine
            out_chunks.append(item)
        return {
            "session_id": sid, "job_id": row["job_id"], "status": row["status"],
            "created_at": row["created_at"], "duration_sec": row["duration_sec"],
            "allowed_languages": stored["allowed_languages"],
            "settings_summary": O.summary(stored["base"]),
            "chunk_count": len(out_chunks),
            "chunks_passed": sum(1 for c in out_chunks if c["passed"]),
            "result_job_id": row["result_job_id"],
            "chunks": out_chunks,
        }

    @staticmethod
    def _best(attempts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Passed attempts first, then externally validated ones, then Whisper's own confidence."""
        if not attempts:
            return None

        def key(a: Dict[str, Any]):
            ext = a.get("external") or {}
            ext_ok = 1 if ext.get("passed") is True else (0 if ext.get("passed") is None else -1)
            lp = a["checks"]["signals"].get("avg_logprob")
            return (1 if a["passed"] else 0, ext_ok, lp if lp is not None else -99, a["created_at"])

        return max(attempts, key=key)

    # --- chunks ----------------------------------------------------------------------------

    def chunk_audio(self, sid: str, chunk_id: int) -> Path:
        row = self._session_row(sid)
        c = self._chunk(sid, chunk_id)
        path = self.root / sid / "chunks" / f"{chunk_id}-{c['start_sec']:.3f}-{c['end_sec']:.3f}.flac"
        if not Path(row["prepared_path"]).exists():
            raise SessionError("This session's audio has expired (kept 7 days); start a new session.", 410)
        if not path.exists():
            extract_segment(row["prepared_path"], c["start_sec"], c["end_sec"], path)
        return path

    def transcribe(self, sid: str, chunk_id: int, params: Optional[Dict[str, Any]] = None,
                   label: Optional[str] = None) -> Dict[str, Any]:
        params = dict(params or {})
        row = self._session_row(sid)
        if row["status"] != "open":
            raise SessionError("This session is already assembled; start a new one to keep working.", 409)
        stored = json.loads(row["params_json"])
        c = self._chunk(sid, chunk_id)
        opts = O.resolve(_params_to_raw(params), base=stored["base"])
        allowed = params.get("allowed_languages") or stored["allowed_languages"]
        if isinstance(allowed, str):
            allowed = [a.strip() for a in allowed.split(",") if a.strip()]
        audio = self.chunk_audio(sid, chunk_id)
        cfg = self.cfg_loader()
        client = self.client_factory(cfg, opts)
        t0 = time.monotonic()
        res = client.transcribe_detailed(audio, lambda _m: None)
        took = int((time.monotonic() - t0) * 1000)
        offset = c["start_sec"]
        segments = [dict(s, start=round(s["start"] + offset, 2), end=round(s["end"] + offset, 2)) for s in res.segments]
        chunk_len = c["end_sec"] - c["start_sec"]
        checks = check_chunk(
            res.text, res.segments, res.language, allowed, dropped=res.dropped,
            compression_ratio_max=opts["compression_ratio_threshold"], no_speech_max=opts["no_speech_threshold"],
            chunk_duration=chunk_len,
        )
        gap = checks["signals"].get("largest_gap")
        if "low_coverage" in checks["problems"] and gap:
            # Split where the uncovered speech starts, so the dropped sentence gets its own decode.
            at = gap["start"] + 0.15 if gap["start"] > 1.0 else gap["end"] - 0.15
            checks["suggested_split"] = round(c["start_sec"] + max(1.0, min(chunk_len - 1.0, at)), 2)
        checks["signals"]["language_probability"] = res.language_probability
        if res.race_scores:
            checks["signals"]["race_scores"] = res.race_scores
        if res.dropped:
            checks["signals"]["dropped"] = [{"text": t[:120], "reason": r} for t, r in res.dropped[:10]]
        attempt_id = uuid.uuid4().hex[:16]
        recorded = {k: v for k, v in params.items() if k != "allowed_languages"}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_attempts (id, session_id, chunk_id, created_at, start_sec, end_sec, params_json, "
                "text, segments_json, language, model, checks_json, passed, duration_ms, label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (attempt_id, sid, chunk_id, utcnow_iso(), c["start_sec"], c["end_sec"], json.dumps(recorded),
                 res.text, json.dumps(segments, ensure_ascii=False), res.language, getattr(client, "model", None),
                 json.dumps(checks, ensure_ascii=False), 1 if checks["passed"] else 0, took, (label or "")[:60] or None),
            )
            arow = conn.execute("SELECT * FROM session_attempts WHERE id = ?", (attempt_id,)).fetchone()
        out = self._attempt_json(arow, c)
        out["segments"] = segments
        out["settings_summary"] = O.summary(opts)
        return out

    def probe(self, sid: str, chunk_id: int, attempt_id: Optional[str] = None,
              min_segment_sec: float = 1.0) -> Dict[str, Any]:
        """Re-detect the language of each segment's own audio.

        Catches the failure text checks cannot see: Whisper decoding a chunk in
        one language and *translating* a sentence spoken in another. A segment
        whose audio is detected as a different allowed language is a mismatch;
        ``suggested_split`` is where to cut so each language gets its own decode.
        """
        row = self._session_row(sid)
        stored = json.loads(row["params_json"])
        c = self._chunk(sid, chunk_id)
        with self._connect() as conn:
            if attempt_id:
                arow = conn.execute("SELECT * FROM session_attempts WHERE id = ? AND session_id = ? AND chunk_id = ?",
                                    (attempt_id, sid, chunk_id)).fetchone()
            else:
                arow = conn.execute("SELECT * FROM session_attempts WHERE session_id = ? AND chunk_id = ? "
                                    "ORDER BY created_at DESC LIMIT 1", (sid, chunk_id)).fetchone()
        if arow is None:
            raise SessionError("No attempt to probe — transcribe the chunk first.", 404)
        attempt_lang = (arow["language"] or "").lower()
        allowed = [a.lower() for a in stored["allowed_languages"]]
        opts = O.resolve({"language_mode": "auto", "beam_size": 1, "best_of": 1, "prompt": "", "hotwords": ""},
                         base=stored["base"])
        client = self.client_factory(self.cfg_loader(), opts)
        results, mismatch_at = [], None
        for i, seg in enumerate(json.loads(arow["segments_json"])):
            start, end = float(seg["start"]), float(seg["end"])
            item = {"start": start, "end": end, "text": seg.get("text", "")[:120], "attempt_language": attempt_lang}
            if end - start < min_segment_sec:
                item["skipped"] = "too short to detect reliably"
                results.append(item)
                continue
            path = self.root / sid / "probe" / f"{arow['id']}-{i}.flac"
            if not path.exists():
                extract_segment(row["prepared_path"], start, end, path)
            res = client.transcribe_detailed(path, lambda _m: None)
            detected = (res.language or "").lower() or None
            item.update(detected_language=detected, detected_text=(res.text or "")[:160],
                        mismatch=bool(detected and detected != attempt_lang and detected in allowed))
            if item["mismatch"] and mismatch_at is None:
                mismatch_at = start if start - c["start_sec"] >= 1.0 else end
            results.append(item)
        out = {"chunk_id": chunk_id, "attempt_id": arow["id"], "attempt_language": attempt_lang,
               "segments": results, "mismatch": any(r.get("mismatch") for r in results)}
        if mismatch_at is not None and c["start_sec"] + 1.0 <= mismatch_at <= c["end_sec"] - 1.0:
            out["suggested_split"] = round(mismatch_at, 2)
        return out

    def record_validation(self, sid: str, attempt_id: str, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Store an outside validator's verdict (e.g. jev-1.13) on an attempt.

        ``verdict`` is free-form; its ``passed`` (bool) is used when picking the best attempt.
        """
        self._session_row(sid)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_attempts WHERE id = ? AND session_id = ?", (attempt_id, sid)
            ).fetchone()
            if row is None:
                raise SessionError("Unknown attempt.", 404)
            conn.execute("UPDATE session_attempts SET external_json = ? WHERE id = ?",
                         (json.dumps(verdict, ensure_ascii=False)[:20000], attempt_id))
            row = conn.execute("SELECT * FROM session_attempts WHERE id = ?", (attempt_id,)).fetchone()
        return self._attempt_json(row, self._chunk(sid, row["chunk_id"]))

    def attempts(self, sid: str, chunk_id: int) -> List[Dict[str, Any]]:
        c = self._chunk(sid, chunk_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_attempts WHERE session_id = ? AND chunk_id = ? ORDER BY created_at",
                (sid, chunk_id),
            ).fetchall()
        return [self._attempt_json(r, c) for r in rows]

    def recut(self, sid: str, chunk_id: int, start: float, end: float) -> Dict[str, Any]:
        """Move a chunk's edges; the neighbours' touching edges move with it (no gaps, no overlap)."""
        row = self._session_row(sid)
        chunks = self._chunks(sid)
        ids = [c["chunk_id"] for c in chunks]
        if chunk_id not in ids:
            raise SessionError(f"Chunk {chunk_id} is not in this session.", 404)
        i = ids.index(chunk_id)
        prev = chunks[i - 1] if i > 0 else None
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        lo = prev["start_sec"] + 0.5 if prev else 0.0
        hi = nxt["end_sec"] - 0.5 if nxt else float(row["duration_sec"])
        start, end = round(float(start), 3), round(float(end), 3)
        if not (lo - 1e-6 <= start < end <= hi + 1e-6) or end - start < 1.0:
            raise SessionError(f"New edges must stay within {lo:.2f}–{hi:.2f} s and span at least 1 s.")
        if prev is None:
            start = 0.0
        if nxt is None:
            end = float(row["duration_sec"])
        with self._connect() as conn:
            conn.execute("UPDATE session_chunks SET start_sec = ?, end_sec = ? WHERE session_id = ? AND chunk_id = ?",
                         (start, end, sid, chunk_id))
            if prev is not None:
                conn.execute("UPDATE session_chunks SET end_sec = ? WHERE session_id = ? AND chunk_id = ?",
                             (start, sid, prev["chunk_id"]))
            if nxt is not None:
                conn.execute("UPDATE session_chunks SET start_sec = ? WHERE session_id = ? AND chunk_id = ?",
                             (end, sid, nxt["chunk_id"]))
        return self.get(sid)

    def split(self, sid: str, chunk_id: int, at: Optional[float] = None) -> Dict[str, Any]:
        """Split a chunk in two at ``at`` (default: its middle). The second half gets a new id."""
        self._session_row(sid)
        c = self._chunk(sid, chunk_id)
        at = round(float(at if at is not None else (c["start_sec"] + c["end_sec"]) / 2), 3)
        if not (c["start_sec"] + 1.0 <= at <= c["end_sec"] - 1.0):
            raise SessionError("Split point must leave at least 1 s on each side.")
        chunks = self._chunks(sid)
        ids = [x["chunk_id"] for x in chunks]
        i = ids.index(chunk_id)
        nxt_pos = chunks[i + 1]["position"] if i + 1 < len(chunks) else c["position"] + 1
        with self._connect() as conn:
            new_id = conn.execute(
                "SELECT COALESCE(MAX(chunk_id), -1) + 1 FROM session_chunks WHERE session_id = ?", (sid,)
            ).fetchone()[0]
            conn.execute("UPDATE session_chunks SET end_sec = ? WHERE session_id = ? AND chunk_id = ?",
                         (at, sid, chunk_id))
            conn.execute(
                "INSERT INTO session_chunks (session_id, chunk_id, position, start_sec, end_sec) VALUES (?, ?, ?, ?, ?)",
                (sid, new_id, (c["position"] + nxt_pos) / 2, at, c["end_sec"]),
            )
        out = self.get(sid)
        out["new_chunk_id"] = new_id
        return out

    # --- assemble --------------------------------------------------------------------------

    def assemble(self, sid: str, choices: Optional[Dict[str, str]] = None, fallback: str = "best",
                 report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write the chosen attempts as a library transcript and return the job.

        ``choices`` maps chunk_id → attempt_id; chunks without a choice use their
        best attempt (``fallback="best"``) or fail (``fallback="error"``).
        """
        row = self._session_row(sid)
        if row["status"] != "open":
            raise SessionError("This session is already assembled.", 409)
        choices = {int(k): v for k, v in (choices or {}).items()}
        chunks = self._chunks(sid)
        with self._connect() as conn:
            arows = conn.execute("SELECT * FROM session_attempts WHERE session_id = ?", (sid,)).fetchall()
        by_id = {a["id"]: a for a in arows}
        picked: List[tuple] = []
        missing, needs_review = [], []
        for c in chunks:
            cid = c["chunk_id"]
            fresh = [self._attempt_json(a, c) for a in arows if a["chunk_id"] == cid]
            fresh = [a for a in fresh if not a["stale"]]
            if cid in choices:
                a = by_id.get(choices[cid])
                if a is None or a["chunk_id"] != cid:
                    raise SessionError(f"Attempt {choices[cid]} does not belong to chunk {cid}.")
                chosen = self._attempt_json(a, c)
                if chosen["stale"]:
                    raise SessionError(f"Attempt {choices[cid]} was made before chunk {cid} was re-cut.")
            elif fallback == "best" and fresh:
                chosen = self._best(fresh)
            else:
                missing.append(cid)
                continue
            if not chosen["passed"]:
                needs_review.append(cid)
            picked.append((c, chosen, by_id[chosen["attempt_id"]]))
        if missing:
            raise SessionError(f"No usable attempt for chunk(s): {', '.join(map(str, missing))}.")

        text_parts: List[str] = []
        segments: List[Dict[str, Any]] = []
        tally: Counter = Counter()
        models: Counter = Counter()
        prev_lang, prev_end = None, None
        for c, chosen, arow in picked:
            segs = json.loads(arow["segments_json"])
            t = (chosen["text"] or "").strip()
            if t:
                gap = (segs[0]["start"] - prev_end) if (segs and prev_end is not None) else 0
                new_par = not text_parts or (chosen["language"] and chosen["language"] != prev_lang) or gap > PARAGRAPH_GAP_SEC
                if new_par:
                    text_parts.append(t)
                else:
                    text_parts[-1] += " " + t
                prev_lang = chosen["language"] or prev_lang
            if segs:
                prev_end = segs[-1]["end"]
            for s in segs:
                s = dict(s)
                if chosen["language"]:
                    s["language"] = chosen["language"]
                segments.append(s)
                if chosen["language"]:
                    tally[chosen["language"]] += max(0.0, s["end"] - s["start"])
            if chosen["model"]:
                models[chosen["model"]] += 1

        stored = json.loads(row["params_json"])
        job = self.storage.get_job(row["job_id"])
        opts = dict(stored["base"])
        opts["pipeline"] = "validated"
        opts["session_id"] = sid
        opts["_summary"] = (f"validated per chunk ({len(picked)} chunks, "
                            f"{len(picked) - len(needs_review)} passed) · {O.summary(stored['base'])}")
        full_report = {
            "session_id": sid,
            "chunks": [{"chunk_id": c["chunk_id"], "start": c["start_sec"], "end": c["end_sec"],
                        "attempt_id": ch["attempt_id"], "label": ch["label"], "passed": ch["passed"],
                        "language": ch["language"], "problems": ch["checks"]["problems"],
                        "external": ch["external"]} for c, ch, _ in picked],
            "needs_review": needs_review,
            "caller": report or {},
        }
        if job.status in ("received", "error", "interrupted") and not job.transcript_path:
            target_id = job.id
        else:
            family = self.storage.versions_of(job.id)
            target_id = uuid.uuid4().hex
            self.storage.create_job(
                target_id, job.original_name, utcnow_iso(), parent_id=job.id,
                version=max((j.version or 1) for j in family) + 1,
                source=job.source, source_detail=job.source_detail, recorded_at=job.recorded_at,
                file_size=job.file_size, file_sha256=job.file_sha256, audio_path=job.audio_path,
            )
        self.storage.update_job(target_id, settings_json=json.dumps(opts))
        data_dir = self.jobs.data_dir
        tdir = data_dir / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{target_id}.txt").write_text("\n\n".join(text_parts), encoding="utf-8")
        sdir = data_dir / "segments"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / f"{target_id}.json").write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        rdir = data_dir / "reports"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"{target_id}.json").write_text(json.dumps(full_report, ensure_ascii=False, indent=1), encoding="utf-8")
        self.storage.update_job(
            target_id, transcript_path=f"transcripts/{target_id}.txt", segments_path=f"segments/{target_id}.json",
            duration_sec=row["duration_sec"], languages=",".join(l for l, _ in tally.most_common()),
            provider="local", model=(models.most_common(1)[0][0] if models else None), error=None,
        )
        self.storage.update_status(target_id, "done", completed_at=utcnow_iso())
        self.jobs._append_log(target_id, f"Assembled from validated session {sid}: {len(picked)} chunks, "
                                         f"{len(needs_review)} need review.")
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET status = 'assembled', result_job_id = ? WHERE id = ?", (target_id, sid))
        self.jobs._after_transcription(target_id, opts)
        return {"job_id": target_id, "chunks": len(picked), "needs_review": needs_review,
                "languages": [l for l, _ in tally.most_common()], "session_id": sid}


def purge_old_sessions(root: Path, max_age_days: int = 7, now: Optional[float] = None) -> int:
    """Drop session audio (prepared + chunk files) after a week; attempts stay in the database."""
    now = now or time.time()
    if not root.exists():
        return 0
    removed = 0
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < now - max_age_days * 86400:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
