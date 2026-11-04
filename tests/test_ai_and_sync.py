import json
from types import SimpleNamespace

import httpx
import pytest

from web import ai as A
from web import supabase_sync as S


# --- AI guard -----------------------------------------------------------------------------

ORIG = ("Ну, коллеги, э, по Skyscanner ситуация следующая. Стоимость лида снизилась до тридцати долларов, "
        "но, ну, количество продаж осталось прежним. Thank you for watching.")


def test_guard_accepts_a_faithful_cleanup():
    clean = ("Коллеги, по Skyscanner ситуация следующая. Стоимость лида снизилась до тридцати долларов, "
             "но количество продаж осталось прежним.")
    assert A.guard(ORIG, clean) is None


def test_guard_rejects_translation_summary_and_invention():
    assert "introduced new words" in A.guard(ORIG, "Colleagues, on Skyscanner the lead cost fell to thirty dollars "
                                                   "but sales stayed flat, thank you for watching the call.")
    assert "too short" in A.guard(ORIG, "Коллеги, по Skyscanner ситуация следующая.")
    assert "introduced new words" in A.guard(ORIG, "Лиды подешевели.")  # a paraphrase, not a cleanup
    assert "too long" in A.guard(ORIG, ORIG + " " + ORIG)
    assert A.guard(ORIG, "  ") == "empty output"


def test_split_text_respects_limit_and_order():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 300 for i in range(10))
    parts = A.split_text(text, limit=2000)
    assert all(len(p) <= 2000 for p in parts)
    assert " ".join(parts).split()[:2] == ["Paragraph", "0."]
    assert A.split_text("") == []


def test_parse_json_object_tolerates_fences_and_chatter():
    assert A.parse_json_object('Sure!\n```json\n{"title": "A {b}"}\n```') == {"title": "A {b}"}
    assert A.parse_json_object('x {"a": {"b": 1}} y') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        A.parse_json_object("no json here")


class FakeChat:
    """Cleans by dropping fillers; answers metadata with JSON (first time garbage)."""

    def __init__(self):
        self.calls = 0
        self.completions = self

    def create(self, model, messages):
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if system == A.CLEAN_SYSTEM:
            out = user.replace("Ну, ", "").replace(", э,", ",").replace(", ну,", ",").replace(" Thank you for watching.", "")
            if "BADCHUNK" in user:
                out = "Completely different English text that summarises it."
        elif self.calls < 4 and "retry-me" in user:
            out = "not json"
        else:
            out = json.dumps({"title": "Skyscanner lead cost review.", "description": "Lead cost fell; sales flat.",
                              "tags": ["Skyscanner", "Leads"], "language": "ru", "people": [], "topics": ["cost"]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


def test_finisher_cleans_titles_and_keeps_bad_chunks_verbatim():
    chat = FakeChat()
    fin = A.AIFinisher(A.AIConfig("u", "k", "m", chunk_chars=200), client=SimpleNamespace(chat=chat))
    text = ORIG + "\n\n" + "BADCHUNK " + ORIG
    res = fin.finish(text)
    assert "Thank you for watching" not in res.clean_text.split("\n\n")[0]
    assert res.clean_text.split("\n\n")[1].startswith("BADCHUNK Ну,")  # guard kept the original
    assert res.rejected_chunks == 1 and res.kept_chunks == 1
    assert res.title == "Skyscanner lead cost review" and res.tags == ["skyscanner", "leads"]


# --- Supabase ------------------------------------------------------------------------------

def test_setup_sql_has_primary_key_for_upsert():
    sql = S.setup_sql("transcripts")
    assert "id                text primary key" in sql and "enable row level security" in sql


def test_backoff_doubles_and_caps():
    assert [S.backoff_delay(n).total_seconds() for n in (1, 2, 3, 10)] == [60, 120, 240, 3600]


def test_upsert_request_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    S.upsert(S.SupabaseConfig("https://p.supabase.co/", "KEY"), [{"id": "a", "title": "Т"}], client=client)
    assert seen["url"] == "https://p.supabase.co/rest/v1/transcripts?on_conflict=id"
    assert seen["headers"]["apikey"] == "KEY" and seen["headers"]["authorization"] == "Bearer KEY"
    assert "merge-duplicates" in seen["headers"]["prefer"]
    assert seen["body"] == [{"id": "a", "title": "Т"}]


def test_upsert_raises_on_http_error_and_check_explains():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"code": "PGRST205"})))
    with pytest.raises(S.SyncError):
        S.upsert(S.SupabaseConfig("https://p", "k"), [{"id": "a"}], client=client)
    ok, msg = S.check(S.SupabaseConfig("https://p", "k"), client=client)
    assert not ok and "does not exist" in msg
    bad_key = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    assert "service_role" in S.check(S.SupabaseConfig("https://p", "k"), client=bad_key)[1]


def test_sync_waits_for_settings_then_delivers_and_backs_off(make_app, settings):
    sent, fail = [], {"n": 1}

    def fake_upsert(cfg, rows):
        if fail["n"] > 0:
            fail["n"] -= 1
            raise S.SyncError("HTTP 503: busy")
        sent.extend(rows)

    c = make_app(sync_fn=fake_upsert)
    c.storage.create_job("s1", "Call.m4a", "2026-09-30T10:00:00+00:00",
                         settings_json=json.dumps({"sync_supabase": True, "ai_finish": False}))
    (settings.uploads_dir / "s1").mkdir()
    (settings.uploads_dir / "s1" / "Call.m4a").write_bytes(b"x")
    c.jobs._run("s1")
    assert c.storage.get_job("s1").sync_status == "pending"  # queued automatically after transcription

    assert c.jobs.sync_once() == 0
    assert c.storage.get_job("s1").sync_error == "Waiting for Supabase settings"

    c.conns.update({"supabase_url": "https://p.supabase.co", "supabase_key": "k"})
    assert c.jobs.sync_once() == 0  # first attempt fails
    job = c.storage.get_job("s1")
    assert job.sync_attempts == 1 and job.sync_next_at and "503" in job.sync_error
    assert c.jobs.sync_once() == 0  # not due yet: backoff respected
    c.storage.update_job("s1", sync_next_at="2000-01-01T00:00:00+00:00")
    assert c.jobs.sync_once() == 1
    job = c.storage.get_job("s1")
    assert job.sync_status == "synced" and job.sync_error is None
    row = sent[0]
    assert row["id"] == "s1" and row["root_id"] == "s1" and row["transcript_raw"] == "the transcript"
    assert row["segments"][0]["text"] == "the transcript" and row["languages"] == ["en"]


def test_ai_pass_resyncs_the_row_with_its_title(make_app, settings):
    sent = []
    chat = FakeChat()
    c = make_app(ai_factory=lambda conns: A.AIFinisher(A.AIConfig("u", "k", "m"), client=SimpleNamespace(chat=chat)),
                 sync_fn=lambda cfg, rows: sent.extend(rows))
    c.conns.update({"openai_api_key": "sk-x", "supabase_url": "https://p", "supabase_key": "k"})
    c.storage.create_job("a1", "Call.m4a", "2026-09-30T10:00:00+00:00",
                         settings_json=json.dumps({"sync_supabase": True, "ai_finish": True}))
    (settings.uploads_dir / "a1").mkdir()
    (settings.uploads_dir / "a1" / "Call.m4a").write_bytes(b"x")
    c.jobs._run("a1")
    assert c.storage.get_job("a1").ai_status == "queued"
    c.jobs._ai_run("a1")  # run the AI worker inline
    job = c.storage.get_job("a1")
    assert job.ai_status == "done" and job.title == "Skyscanner lead cost review"
    assert job.sync_status == "pending"
    c.jobs.sync_once()
    assert sent[-1]["title"] == "Skyscanner lead cost review" and sent[-1]["transcript_clean"] is not None
    assert sent[-1]["topics"] == ["cost"]
