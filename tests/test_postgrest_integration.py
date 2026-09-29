"""Delivery against a REAL PostgREST (the engine behind Supabase's REST API).

Skipped unless VT_POSTGREST_URL and VT_POSTGREST_JWT point at a PostgREST whose
database has the setup SQL applied (see deploy notes). Proves the row shape,
text[] / jsonb / timestamptz columns and upsert-on-id semantics end to end.
"""
import json
import os

import httpx
import pytest

from web import supabase_sync as S
from web.storage import Storage

URL, JWT = os.getenv("VT_POSTGREST_URL"), os.getenv("VT_POSTGREST_JWT")
pytestmark = pytest.mark.skipif(not (URL and JWT), reason="no PostgREST configured")


def cfg(table="transcripts"):
    return S.SupabaseConfig(URL, JWT, table, rest_path="")


def test_check_and_upsert_roundtrip(settings):
    assert S.check(cfg()) == (True, "Connected. Table 'transcripts' is ready.")
    ok, msg = S.check(cfg("nope_table"))
    assert not ok and "does not exist" in msg

    store = Storage(settings.db_path)
    store.create_job("it1", "Встреча.m4a", "2026-09-30T10:00:00+00:00",
                     settings_json=json.dumps({"_summary": "large-v3", "beam_size": 5}),
                     recorded_at="2026-09-29T10:15:00", file_sha256="f" * 64, file_size=123,
                     languages="ru,en", tags=json.dumps(["kayak", "бюджет"]), source="phone",
                     source_detail="S23 phone")
    store.update_status("it1", "done", completed_at="2026-09-30T10:01:00+00:00")
    job = store.get_job("it1")
    segs = [{"start": 0.0, "end": 1.5, "text": "Добрый день", "language": "ru"}]
    row = S.build_row(job, "Добрый день", None, segs, root_id="it1", ai_meta={"people": ["Eldar"]})
    S.upsert(cfg(), [row])
    # second delivery (e.g. after the AI pass) updates the same row
    row2 = dict(row, title="Budget sync", transcript_clean="Добрый день.", ai_model="gpt-x")
    S.upsert(cfg(), [row2])

    got = httpx.get(f"{URL}/transcripts", params={"id": "eq.it1"},
                    headers={"Authorization": f"Bearer {JWT}"}).json()
    assert len(got) == 1
    r = got[0]
    assert r["title"] == "Budget sync" and r["transcript_clean"] == "Добрый день."
    assert r["languages"] == ["ru", "en"] and r["tags"] == ["kayak", "бюджет"] and r["people"] == ["Eldar"]
    assert r["segments"][0]["text"] == "Добрый день" and r["settings"]["beam_size"] == 5
    assert r["recorded_at"].startswith("2026-09-29T10:15:00") and r["file_size"] == 123
    assert r["source"] == "phone" and r["root_id"] == "it1" and r["version"] == 1


def test_anonymous_access_is_blocked_by_rls():
    resp = httpx.get(f"{URL}/transcripts")  # no JWT -> anon role, no grants/policies
    assert resp.status_code in (401, 403) or resp.json() == []
