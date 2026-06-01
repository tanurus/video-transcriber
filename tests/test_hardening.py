import pytest

from web import create_app
from web.storage import Storage


class RecordingJobs:
    def __init__(self):
        self.last_since = None

    def submit(self, job_id, video_path):
        pass

    def get_logs(self, job_id, since=0):
        self.last_since = since
        return [], 0


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = RecordingJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    c = app.test_client()
    c._storage = storage
    c._jobs = jobs
    c._settings = settings
    return c


def test_download_rejects_path_escape(client):
    # A transcript_path that escapes the data dir must 404, even if the target exists.
    secret = client._settings.data_dir.parent / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    client._storage.create_job("j", "x.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j", "done", transcript_path="../secret.txt")
    assert client.get("/download/j").status_code == 404


def test_job_api_clamps_negative_since(client):
    client._storage.create_job("j2", "x.mp4", "2026-06-01T00:00:00+00:00")
    resp = client.get("/api/job/j2?since=-5")
    assert resp.status_code == 200
    assert client._jobs.last_since == 0


def test_update_status_rejects_invalid_status(settings):
    store = Storage(settings.db_path)
    store.create_job("j", "x.mp4", "2026-06-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        store.update_status("j", "bogus")


def test_log_buffer_caps_tracked_jobs(settings):
    from web.jobs import JobManager, MAX_TRACKED_JOBS

    store = Storage(settings.db_path)
    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=lambda *a, **k: None)
    for i in range(MAX_TRACKED_JOBS + 10):
        mgr._append_log(f"job{i}", "line")
    assert len(mgr._logs) == MAX_TRACKED_JOBS
    # oldest evicted, newest retained
    assert mgr.get_logs("job0")[0] == []
    assert mgr.get_logs(f"job{MAX_TRACKED_JOBS + 9}")[0] == ["line"]
