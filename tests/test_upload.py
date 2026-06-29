import io

import pytest

from web import create_app
from web.storage import Storage


class FakeJobs:
    def __init__(self):
        self.submitted = []

    def submit(self, job_id, video_path):
        self.submitted.append((job_id, str(video_path)))

    def get_logs(self, job_id, since=0):
        return [], 0


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


def test_upload_valid_creates_job_and_redirects(client):
    data = {"video": (io.BytesIO(b"fake video bytes"), "Meeting.mp4")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/job/" in resp.headers["Location"]
    # exactly one job created and submitted
    jobs = client._storage.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].original_name == "Meeting.mp4"
    assert len(client._jobs.submitted) == 1
    # video saved on disk
    job_id = jobs[0].id
    saved = client._settings.uploads_dir / job_id
    assert any(saved.iterdir())


def test_upload_ajax_returns_json_job_url(client):
    data = {"video": (io.BytesIO(b"fake video bytes"), "Meeting.mp4")}
    resp = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["job_id"]
    assert "/job/" in payload["job_url"]
    jobs = client._storage.list_jobs()
    assert len(jobs) == 1
    assert len(client._jobs.submitted) == 1


def test_upload_ajax_bad_extension_returns_json_error(client):
    data = {"video": (io.BytesIO(b"x"), "notes.txt")}
    resp = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()
    assert client._storage.list_jobs() == []
    assert client._jobs.submitted == []


def test_upload_missing_file_flashes_and_redirects(client):
    resp = client.post("/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert client._storage.list_jobs() == []


def test_upload_bad_extension_rejected(client):
    data = {"video": (io.BytesIO(b"x"), "notes.txt")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert client._storage.list_jobs() == []
    assert client._jobs.submitted == []


def test_upload_non_ascii_filename_keeps_extension(client):
    # secure_filename strips non-ASCII; "Видео.mp4" must not be saved as
    # an extensionless file called "mp4" (the API sniffs format by extension).
    data = {"video": (io.BytesIO(b"x"), "Видео.mp4")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    job_id = client._storage.list_jobs()[0].id
    saved = list((client._settings.uploads_dir / job_id).iterdir())
    assert len(saved) == 1
    assert saved[0].suffix == ".mp4"
