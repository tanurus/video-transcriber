"""Deliver transcripts to a Supabase (PostgREST) table.

Upserts are keyed by the job id, so re-sending — after an AI pass, an edit or
a retry — updates the same row instead of duplicating it.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

SETUP_SQL = """-- Run once in the Supabase SQL editor (or ask Claude to apply it).
create table if not exists public.{table} (
  id                text primary key,            -- the transcriber's job id
  root_id           text not null,               -- first job of this recording (versions share it)
  parent_id         text,                        -- the job this one regenerated
  version           integer not null default 1,
  title             text,
  description       text,
  tags              text[] default '{{}}',
  people            text[] default '{{}}',
  topics            text[] default '{{}}',
  original_filename text not null,
  source            text,                        -- web | api | gui ...
  source_detail     text,                        -- device / user agent / API token name
  recorded_at       timestamptz,                 -- best known recording time
  created_at        timestamptz not null,
  completed_at      timestamptz,
  duration_sec      double precision,
  languages         text[] default '{{}}',
  provider          text,
  model             text,
  settings          jsonb,
  settings_summary  text,
  file_sha256       text,
  file_size         bigint,
  transcript_raw    text,
  transcript_clean  text,
  segments          jsonb,
  ai_model          text,
  host              text,
  synced_at         timestamptz not null default now()
);
create index if not exists {table}_recorded_at_idx on public.{table} (recorded_at desc);
create index if not exists {table}_sha_idx on public.{table} (file_sha256);
create index if not exists {table}_root_idx on public.{table} (root_id);
-- Only the service key (used by the transcriber) can read or write; add policies to expose it.
alter table public.{table} enable row level security;
"""


class SyncError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class SupabaseConfig:
    url: str
    key: str
    table: str = "transcripts"
    timeout: float = 60.0

    def endpoint(self) -> str:
        return f"{self.url.rstrip('/')}/rest/v1/{self.table}"

    def headers(self) -> Dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }


def setup_sql(table: str) -> str:
    return SETUP_SQL.format(table=table)


def backoff_delay(attempts: int) -> timedelta:
    """1 min, 2, 4, ... capped at 1 hour."""
    return timedelta(seconds=min(3600, 60 * (2 ** max(0, attempts - 1))))


def next_attempt_iso(attempts: int, now: Optional[datetime] = None) -> str:
    return ((now or datetime.now(timezone.utc)) + backoff_delay(attempts)).isoformat()


def upsert(cfg: SupabaseConfig, rows: List[Dict[str, Any]], client: Optional[httpx.Client] = None) -> None:
    if not rows:
        return
    headers = cfg.headers() | {"Prefer": "resolution=merge-duplicates,return=minimal"}
    own = client is None
    client = client or httpx.Client(timeout=cfg.timeout)
    try:
        resp = client.post(cfg.endpoint(), params={"on_conflict": "id"}, headers=headers,
                           content=json.dumps(rows, ensure_ascii=False, default=str))
    except httpx.HTTPError as e:
        raise SyncError(f"network error: {e}") from e
    finally:
        if own:
            client.close()
    if resp.status_code >= 300:
        body = resp.text[:500]
        # 4xx (bad key, missing table/column) will not fix itself by retrying fast,
        # but the user can fix the settings/schema — so keep retrying with backoff.
        raise SyncError(f"HTTP {resp.status_code}: {body}", retryable=resp.status_code >= 500 or resp.status_code in (401, 403, 404, 408, 429))


def check(cfg: SupabaseConfig, client: Optional[httpx.Client] = None) -> Tuple[bool, str]:
    """Probe the table; returns (ok, human message)."""
    own = client is None
    client = client or httpx.Client(timeout=15)
    try:
        resp = client.get(cfg.endpoint(), params={"select": "id", "limit": "1"}, headers=cfg.headers())
    except httpx.HTTPError as e:
        return False, f"Cannot reach {cfg.url}: {e}"
    finally:
        if own:
            client.close()
    if resp.status_code == 200:
        return True, f"Connected. Table '{cfg.table}' is ready."
    body = resp.text[:300]
    if resp.status_code == 404 or "PGRST205" in body or "does not exist" in body:
        return False, f"Connected, but table '{cfg.table}' does not exist yet — run the setup SQL below."
    if resp.status_code in (401, 403):
        return False, "The key was rejected. Use the service_role (or sb_secret_…) key."
    return False, f"HTTP {resp.status_code}: {body}"


def _iso_or_none(value: Optional[str]) -> Optional[str]:
    return value or None


def build_row(job: Any, raw_text: str, clean_text: Optional[str], segments: Optional[list],
              root_id: str, ai_meta: Optional[dict] = None) -> Dict[str, Any]:
    ai_meta = ai_meta or {}
    settings = job.settings
    return {
        "id": job.id,
        "root_id": root_id,
        "parent_id": job.parent_id,
        "version": job.version or 1,
        "title": job.title,
        "description": job.description,
        "tags": job.tag_list,
        "people": ai_meta.get("people") or [],
        "topics": ai_meta.get("topics") or [],
        "original_filename": job.original_name,
        "source": job.source,
        "source_detail": job.source_detail,
        "recorded_at": _iso_or_none(job.recorded_at),
        "created_at": job.created_at,
        "completed_at": _iso_or_none(job.completed_at),
        "duration_sec": job.duration_sec,
        "languages": [x for x in (job.languages or "").split(",") if x],
        "provider": job.provider,
        "model": job.model,
        "settings": settings,
        "settings_summary": settings.get("_summary"),
        "file_sha256": job.file_sha256,
        "file_size": job.file_size,
        "transcript_raw": raw_text,
        "transcript_clean": clean_text,
        "segments": segments,
        "ai_model": ai_meta.get("model"),
        "host": socket.gethostname(),
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
