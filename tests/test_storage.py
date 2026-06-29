from web.storage import Storage, Job


def test_create_and_get(settings):
    store = Storage(settings.db_path)
    store.create_job("abc", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    job = store.get_job("abc")
    assert isinstance(job, Job)
    assert job.id == "abc"
    assert job.original_name == "Clip.mp4"
    assert job.status == "queued"
    assert job.completed_at is None


def test_get_missing_returns_none(settings):
    store = Storage(settings.db_path)
    assert store.get_job("nope") is None


def test_update_status_to_done(settings):
    store = Storage(settings.db_path)
    store.create_job("abc", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status(
        "abc", "done",
        completed_at="2026-06-01T00:05:00+00:00",
        transcript_path="transcripts/abc.txt",
    )
    job = store.get_job("abc")
    assert job.status == "done"
    assert job.completed_at == "2026-06-01T00:05:00+00:00"
    assert job.transcript_path == "transcripts/abc.txt"


def test_update_status_to_error(settings):
    store = Storage(settings.db_path)
    store.create_job("e1", "Bad.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("e1", "error", error="boom")
    job = store.get_job("e1")
    assert job.status == "error"
    assert job.error == "boom"


def test_list_jobs_newest_first(settings):
    store = Storage(settings.db_path)
    store.create_job("a", "A.mp4", "2026-06-01T00:00:00+00:00")
    store.create_job("b", "B.mp4", "2026-06-02T00:00:00+00:00")
    ids = [j.id for j in store.list_jobs()]
    assert ids == ["b", "a"]


def test_reconcile_interrupted_only_touches_running(settings):
    # 'queued' jobs are requeued at boot (create_app), so reconcile must leave
    # them alone; only 'running' work was lost with the dead process.
    store = Storage(settings.db_path)
    store.create_job("q", "Q.mp4", "2026-06-01T00:00:00+00:00")  # queued
    store.create_job("r", "R.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("r", "running")
    store.create_job("d", "D.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("d", "done")
    count = store.reconcile_interrupted()
    assert count == 1
    assert store.get_job("q").status == "queued"
    assert store.get_job("r").status == "interrupted"
    assert store.get_job("d").status == "done"
