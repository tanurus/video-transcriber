import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    """Records submit calls; never does real work."""

    def __init__(self):
        self.submitted = []
        self._logs = {}

    def submit(self, job_id, video_path):
        self.submitted.append((job_id, str(video_path)))

    def get_logs(self, job_id, since=0):
        lines = self._logs.get(job_id, [])
        return list(lines[since:]), len(lines)


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    client = app.test_client()
    client._storage = storage  # type: ignore[attr-defined]
    client._jobs = jobs  # type: ignore[attr-defined]
    return client


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert b"ok" in resp.data


def test_index_empty(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Upload" in resp.data


def test_index_lists_jobs(client):
    client._storage.create_job("a", "Alpha.mp4", "2026-06-01T00:00:00+00:00")
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Alpha.mp4" in resp.data
