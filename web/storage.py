from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

VALID_STATUSES = {"queued", "running", "done", "error", "interrupted"}
AI_STATUSES = {"none", "queued", "running", "done", "error"}
SYNC_STATUSES = {"none", "pending", "synced", "error"}

# Columns added after the first release. Each is applied with ALTER TABLE on
# startup when missing, so an existing app.db upgrades in place.
_MIGRATED_COLUMNS: list[tuple[str, str]] = [
    ("title", "TEXT"),
    ("description", "TEXT"),
    ("tags", "TEXT"),  # JSON list
    ("source", "TEXT"),  # web | api | gui | ... (where the recording came from)
    ("source_detail", "TEXT"),  # device / user agent / API client name
    ("recorded_at", "TEXT"),  # best known recording time (file mtime from the browser)
    ("file_size", "INTEGER"),
    ("file_sha256", "TEXT"),
    ("duration_sec", "REAL"),
    ("languages", "TEXT"),  # comma-separated detected languages, most frequent first
    ("provider", "TEXT"),
    ("model", "TEXT"),
    ("settings_json", "TEXT"),  # the full effective settings this job ran with
    ("parent_id", "TEXT"),  # the job this one regenerates (NULL for an original)
    ("version", "INTEGER DEFAULT 1"),
    ("segments_path", "TEXT"),
    ("clean_text_path", "TEXT"),
    ("audio_path", "TEXT"),  # kept lossless audio, so regenerate survives the video purge
    ("ai_status", "TEXT DEFAULT 'none'"),
    ("ai_error", "TEXT"),
    ("ai_at", "TEXT"),
    ("sync_status", "TEXT DEFAULT 'none'"),
    ("sync_error", "TEXT"),
    ("sync_attempts", "INTEGER DEFAULT 0"),
    ("sync_next_at", "TEXT"),
    ("synced_at", "TEXT"),
    ("updated_at", "TEXT"),
]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    original_name: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    transcript_path: Optional[str] = None
    error: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[str] = None
    source: Optional[str] = None
    source_detail: Optional[str] = None
    recorded_at: Optional[str] = None
    file_size: Optional[int] = None
    file_sha256: Optional[str] = None
    duration_sec: Optional[float] = None
    languages: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    settings_json: Optional[str] = None
    parent_id: Optional[str] = None
    version: Optional[int] = 1
    segments_path: Optional[str] = None
    clean_text_path: Optional[str] = None
    audio_path: Optional[str] = None
    ai_status: Optional[str] = "none"
    ai_error: Optional[str] = None
    ai_at: Optional[str] = None
    sync_status: Optional[str] = "none"
    sync_error: Optional[str] = None
    sync_attempts: Optional[int] = 0
    sync_next_at: Optional[str] = None
    synced_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.title or self.original_name

    @property
    def settings(self) -> dict:
        try:
            return json.loads(self.settings_json or "{}")
        except ValueError:
            return {}

    @property
    def tag_list(self) -> list[str]:
        try:
            value = json.loads(self.tags or "[]")
            return [str(t) for t in value] if isinstance(value, list) else []
        except ValueError:
            return []


_JOB_FIELDS = {f.name for f in fields(Job)}


def _row_to_job(row: sqlite3.Row) -> Job:
    keys = set(row.keys())
    return Job(**{name: row[name] for name in _JOB_FIELDS if name in keys})


@dataclass
class Profile:
    id: int
    name: str
    prompt: str = ""
    hotwords: str = ""
    languages: str = ""  # candidate languages, comma-separated
    notes: str = ""
    created_at: str = ""


@dataclass
class ApiToken:
    id: int
    name: str
    token_hash: str
    prefix: str
    created_at: str
    last_used_at: Optional[str] = None


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL lets the worker threads and HTTP reader threads proceed without
        # blocking each other. journal_mode is persisted per-db.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    transcript_path TEXT,
                    error TEXT
                )
                """
            )
            have = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
            for name, decl in _MIGRATED_COLUMNS:
                if name not in have:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_sha ON jobs(file_sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    prompt TEXT NOT NULL DEFAULT '',
                    hotwords TEXT NOT NULL DEFAULT '',
                    languages TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    # --- jobs ----------------------------------------------------------------

    def create_job(
        self, job_id: str, original_name: str, created_at: str, **extra: Any
    ) -> Job:
        unknown = set(extra) - _JOB_FIELDS
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        values = {"id": job_id, "original_name": original_name, "status": "queued",
                  "created_at": created_at, "updated_at": created_at, **extra}
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self._connect() as conn:
            conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", list(values.values()))
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def get_jobs(self, job_ids: Iterable[str]) -> list[Job]:
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            return []
        marks = ", ".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM jobs WHERE id IN ({marks})", ids).fetchall()
        by_id = {r["id"]: _row_to_job(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def list_jobs(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC").fetchall()
        return [_row_to_job(r) for r in rows]

    def search_jobs(
        self,
        text: str = "",
        status: str = "",
        source: str = "",
        sync_status: str = "",
        latest_only: bool = False,
        limit: int = 500,
    ) -> list[Job]:
        where, params = [], []
        if text:
            where.append("(title LIKE ? OR original_name LIKE ? OR description LIKE ? OR tags LIKE ?)")
            like = f"%{text}%"
            params += [like, like, like, like]
        if status:
            where.append("status = ?")
            params.append(status)
        if source:
            where.append("source = ?")
            params.append(source)
        if sync_status == "error":
            # Failed sends keep retrying with backoff, so "failed" = pending with an error.
            where.append("(sync_status = 'error' OR (sync_status = 'pending' AND sync_error IS NOT NULL))")
        elif sync_status:
            where.append("COALESCE(sync_status, 'none') = ?")
            params.append(sync_status)
        if latest_only:
            # Hide a version once a newer regeneration of the same recording exists.
            where.append("NOT EXISTS (SELECT 1 FROM jobs c WHERE c.parent_id = jobs.id)")
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_job(r) for r in rows]

    def find_by_sha(self, sha256: str) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE file_sha256 = ? ORDER BY created_at", (sha256,)
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def versions_of(self, job_id: str) -> list[Job]:
        """Every job in the same regeneration family, oldest first."""
        root = self.root_of(job_id)
        family, frontier = [root], [root]
        with self._connect() as conn:
            while frontier:
                marks = ", ".join("?" for _ in frontier)
                rows = conn.execute(
                    f"SELECT id FROM jobs WHERE parent_id IN ({marks})", frontier
                ).fetchall()
                frontier = [r["id"] for r in rows if r["id"] not in family]
                family += frontier
        return sorted(self.get_jobs(family), key=lambda j: (j.version or 1, j.created_at))

    def root_of(self, job_id: str) -> str:
        seen = set()
        current = job_id
        with self._connect() as conn:
            while current not in seen:
                seen.add(current)
                row = conn.execute("SELECT parent_id FROM jobs WHERE id = ?", (current,)).fetchone()
                if row is None or not row["parent_id"]:
                    return current
                current = row["parent_id"]
        return current

    def update_job(self, job_id: str, **values: Any) -> None:
        unknown = set(values) - _JOB_FIELDS
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        if "status" in values and values["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid status: {values['status']!r}")
        if "ai_status" in values and values["ai_status"] not in AI_STATUSES:
            raise ValueError(f"invalid ai_status: {values['ai_status']!r}")
        if "sync_status" in values and values["sync_status"] not in SYNC_STATUSES:
            raise ValueError(f"invalid sync_status: {values['sync_status']!r}")
        if not values:
            return
        values.setdefault("updated_at", utcnow_iso())
        sets = ", ".join(f"{k} = ?" for k in values)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*values.values(), job_id])

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        completed_at: Optional[str] = None,
        transcript_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        values: dict[str, Any] = {"status": status}
        if completed_at is not None:
            values["completed_at"] = completed_at
        if transcript_path is not None:
            values["transcript_path"] = transcript_path
        if error is not None:
            values["error"] = error
        self.update_job(job_id, **values)

    def delete_job(self, job_id: str) -> None:
        with self._connect() as conn:
            # Children keep existing but lose their parent link rather than dangling.
            conn.execute("UPDATE jobs SET parent_id = NULL WHERE parent_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def jobs_with(self, **equals: Any) -> list[Job]:
        where = " AND ".join(f"COALESCE({k}, '') = ?" for k in equals)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE {where} ORDER BY created_at", list(equals.values())
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def due_for_sync(self, now_iso: str, limit: int = 20) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE sync_status = 'pending' "
                "AND (sync_next_at IS NULL OR sync_next_at <= ?) ORDER BY created_at LIMIT ?",
                (now_iso, limit),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def counts(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  SUM(status IN ('queued', 'running')) AS active,
                  SUM(status = 'error') AS failed,
                  SUM(COALESCE(ai_status, 'none') IN ('queued', 'running')) AS ai_active,
                  SUM(sync_status = 'pending') AS sync_pending,
                  SUM(sync_status = 'error') AS sync_failed,
                  COUNT(*) AS total
                FROM jobs
                """
            ).fetchone()
        return {k: int(row[k] or 0) for k in row.keys()}

    def reconcile_interrupted(self) -> int:
        """Mark jobs that were mid-transcription when the process died.

        Only 'running' jobs are interrupted: their partial work is lost. Jobs
        still 'queued' are left untouched so the app can requeue them at boot.
        An AI pass that was running goes back to 'queued' — it is idempotent.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'interrupted' WHERE status = 'running'"
            )
            conn.execute("UPDATE jobs SET ai_status = 'queued' WHERE ai_status = 'running'")
            return cur.rowcount

    # --- profiles --------------------------------------------------------------

    def list_profiles(self) -> list[Profile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY name COLLATE NOCASE").fetchall()
        return [Profile(**dict(r)) for r in rows]

    def get_profile(self, profile_id: int) -> Optional[Profile]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return Profile(**dict(row)) if row else None

    def save_profile(
        self, name: str, prompt: str = "", hotwords: str = "", languages: str = "",
        notes: str = "", profile_id: Optional[int] = None,
    ) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Profile name is required.")
        with self._connect() as conn:
            if profile_id:
                conn.execute(
                    "UPDATE profiles SET name=?, prompt=?, hotwords=?, languages=?, notes=? WHERE id=?",
                    (name, prompt, hotwords, languages, notes, profile_id),
                )
                return int(profile_id)
            cur = conn.execute(
                "INSERT INTO profiles (name, prompt, hotwords, languages, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, prompt, hotwords, languages, notes, utcnow_iso()),
            )
            return int(cur.lastrowid)

    def delete_profile(self, profile_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))

    # --- api tokens --------------------------------------------------------------

    def add_token(self, name: str, token_hash: str, prefix: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO api_tokens (name, token_hash, prefix, created_at) VALUES (?, ?, ?, ?)",
                (name, token_hash, prefix, utcnow_iso()),
            )
            return int(cur.lastrowid)

    def list_tokens(self) -> list[ApiToken]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM api_tokens ORDER BY created_at DESC").fetchall()
        return [ApiToken(**dict(r)) for r in rows]

    def token_by_hash(self, token_hash: str) -> Optional[ApiToken]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM api_tokens WHERE token_hash = ?", (token_hash,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (utcnow_iso(), row["id"])
                )
        return ApiToken(**dict(row)) if row else None

    def delete_token(self, token_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))

    # --- app settings (non-secret) --------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
