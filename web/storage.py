from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALID_STATUSES = {"queued", "running", "done", "error", "interrupted"}


@dataclass
class Job:
    id: str
    original_name: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    transcript_path: Optional[str] = None
    error: Optional[str] = None


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        original_name=row["original_name"],
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        transcript_path=row["transcript_path"],
        error=row["error"],
    )


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # WAL lets the single writer (worker thread) and HTTP reader threads
        # proceed without blocking each other. journal_mode is persisted per-db.
        conn.execute("PRAGMA journal_mode=WAL")
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

    def create_job(self, job_id: str, original_name: str, created_at: str) -> Job:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, original_name, status, created_at) "
                "VALUES (?, ?, 'queued', ?)",
                (job_id, original_name, created_at),
            )
        return Job(id=job_id, original_name=original_name, status="queued", created_at=created_at)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC").fetchall()
        return [_row_to_job(r) for r in rows]

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
        sets = ["status = ?"]
        params: list[object] = [status]
        if completed_at is not None:
            sets.append("completed_at = ?")
            params.append(completed_at)
        if transcript_path is not None:
            sets.append("transcript_path = ?")
            params.append(transcript_path)
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        params.append(job_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)

    def reconcile_interrupted(self) -> int:
        """Mark jobs that were mid-transcription when the process died.

        Only 'running' jobs are interrupted: their partial work is lost. Jobs
        still 'queued' are left untouched so the app can requeue them at boot.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'interrupted' WHERE status = 'running'"
            )
            return cur.rowcount
