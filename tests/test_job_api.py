import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    def __init__(self):
        self._logs = {}

    def submit(self, job_id, video_path):
        pass

    def set_logs(self, job_id, lines):
        self._logs[job_id] = lines

    def get_logs(self, job_id, since=0):
        lines = self._logs.get(job_id, [])
        return list(lines[since:]), len(lines)


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    c = app.test_client()
    c._storage = storage
    c._jobs = jobs
    c._settings = settings
    return c


def test_job_page_renders(client):
    client._storage.create_job("j1", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    resp = client.get("/job/j1")
    assert resp.status_code == 200
    assert b"Clip.mp4" in resp.data


def test_job_page_404(client):
    assert client.get("/job/missing").status_code == 404


def test_job_api_streams_status_and_lines(client):
    client._storage.create_job("j1", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j1", "running")
    client._jobs.set_logs("j1", ["one", "two", "three"])
    resp = client.get("/api/job/j1?since=1")
    body = resp.get_json()
    assert body["status"] == "running"
    assert body["lines"] == ["two", "three"]
    assert body["next_index"] == 3
    assert body["download_ready"] is False


def test_job_api_download_ready_when_done(client):
    client._storage.create_job("j2", "Clip.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j2", "done", transcript_path="transcripts/j2.txt")
    body = client.get("/api/job/j2").get_json()
    assert body["download_ready"] is True


def test_download_serves_transcript(client):
    (client._settings.transcripts_dir / "j3.txt").write_text("the transcript", encoding="utf-8")
    client._storage.create_job("j3", "My Meeting.mp4", "2026-06-01T00:00:00+00:00")
    client._storage.update_status("j3", "done", transcript_path="transcripts/j3.txt")
    resp = client.get("/download/j3")
    assert resp.status_code == 200
    assert resp.data == b"the transcript"
    assert "attachment" in resp.headers["Content-Disposition"]
    assert "My Meeting.txt" in resp.headers["Content-Disposition"]


def test_download_404_when_no_transcript(client):
    client._storage.create_job("j4", "x.mp4", "2026-06-01T00:00:00+00:00")
    assert client.get("/download/j4").status_code == 404
