import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    def submit(self, job_id, video_path):
        pass

    def get_logs(self, job_id, since=0):
        return [], 0


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    app = create_app(settings=settings, storage=storage, job_manager=FakeJobs(), start_background=False)
    app.config.update(TESTING=True)
    c = app.test_client()
    c._storage = storage
    return c


def test_history_lists_all_jobs(client):
    client._storage.create_job("a", "Alpha.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.create_job("b", "Beta.mp4", "2026-06-02T00:00:00+00:00")
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"Alpha.mp4" in resp.data
    assert b"Beta.mp4" in resp.data


def test_history_empty(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"None yet" in resp.data
