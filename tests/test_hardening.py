import pytest

from web.jobs import MAX_TRACKED_JOBS
from web.storage import Storage


def test_download_rejects_path_escape_even_if_target_exists(client, settings):
    # A transcript_path that escapes the data dir must 404, even if the target exists.
    secret = settings.data_dir.parent / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    client.storage.create_job("j", "x.mp4", "2026-06-01T00:00:00+00:00")
    client.storage.update_status("j", "done", transcript_path="../secret.txt")
    assert client.get("/download/j?kind=raw").status_code == 404


def test_update_status_rejects_invalid_status(settings):
    store = Storage(settings.db_path)
    store.create_job("j", "x.mp4", "2026-06-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        store.update_status("j", "bogus")
    with pytest.raises(ValueError):
        store.update_job("j", sync_status="bogus")
    with pytest.raises(ValueError):
        store.update_job("j", not_a_column=1)


def test_log_buffer_caps_memory_but_keeps_the_file(client):
    mgr = client.jobs
    for i in range(MAX_TRACKED_JOBS + 10):
        mgr._append_log(f"job{i}", "line")
    assert len(mgr._logs) == MAX_TRACKED_JOBS
    assert "job0" not in mgr._logs
    assert mgr.get_logs("job0")[0] == ["line"]  # evicted from memory, still on disk


def test_migrates_an_old_database_in_place(settings):
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, original_name TEXT NOT NULL, status TEXT NOT NULL, "
                 "created_at TEXT NOT NULL, completed_at TEXT, transcript_path TEXT, error TEXT)")
    conn.execute("INSERT INTO jobs VALUES ('old', 'v1.mp4', 'done', '2026-09-01T00:00:00', NULL, 't.txt', NULL)")
    conn.commit()
    conn.close()
    store = Storage(settings.db_path)
    job = store.get_job("old")
    assert job.original_name == "v1.mp4" and job.ai_status == "none" and job.version == 1
