import io
import json

import pytest

from app.config import Config
from web import create_app
from web.jobs import FAST_MODEL, JobManager, apply_options, read_options
from web.storage import Storage


class FakeJobs:
    def __init__(self):
        self.submitted = []

    def submit(self, job_id, video_path):
        self.submitted.append((job_id, video_path))

    def get_logs(self, job_id, since=0):
        return [], 0


@pytest.fixture
def client(settings):
    storage = Storage(settings.db_path)
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    c = app.test_client()
    c._jobs = jobs
    return c


def _upload(client, **form):
    data = {"video": (io.BytesIO(b"v"), "Meeting.mp4")}
    data.update(form)
    return client.post("/upload", data=data, content_type="multipart/form-data")


def test_fast_quality_is_persisted_next_to_the_upload(client):
    _upload(client, quality="fast")
    _, video_path = client._jobs.submitted[0]
    assert read_options(video_path) == {"quality": "fast"}


def test_default_quality_writes_no_options_file(client):
    _upload(client)
    _, video_path = client._jobs.submitted[0]
    assert read_options(video_path) == {}
    assert not (video_path.parent / ".options.json").exists()


def test_bogus_quality_is_ignored(client):
    _upload(client, quality="ultra")
    _, video_path = client._jobs.submitted[0]
    assert read_options(video_path) == {}


def test_requeue_skips_the_options_file(settings):
    # ".options.json" sorts before "Meeting.mp4"; requeue must still pick the video.
    storage = Storage(settings.db_path)
    storage.create_job("j1", "Meeting.mp4", "2026-09-30T00:00:00+00:00")
    job_dir = settings.uploads_dir / "j1"
    job_dir.mkdir(parents=True)
    (job_dir / "Meeting.mp4").write_bytes(b"v")
    (job_dir / ".options.json").write_text(json.dumps({"quality": "fast"}))
    jobs = FakeJobs()
    create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    assert [p.name for _, p in jobs.submitted] == ["Meeting.mp4"]


@pytest.mark.parametrize("provider", ["local", "groq"])
def test_fast_switches_whisper_providers_to_turbo(provider):
    cfg = Config(openai_api_key="k", model="whisper-large-v3", provider=provider)
    note = apply_options(cfg, {"quality": "fast"})
    assert cfg.model == FAST_MODEL == "whisper-large-v3-turbo"
    assert "fast" in note.lower()


def test_fast_is_ignored_on_openai():
    cfg = Config(openai_api_key="k", model="gpt-4o-transcribe", provider="openai")
    note = apply_options(cfg, {"quality": "fast"})
    assert cfg.model == "gpt-4o-transcribe"
    assert "not available" in note


def test_no_options_changes_nothing():
    cfg = Config(openai_api_key="k", model="whisper-large-v3", provider="local")
    assert apply_options(cfg, {}) is None
    assert cfg.model == "whisper-large-v3"


def test_job_run_applies_options_before_transcribing(settings):
    storage = Storage(settings.db_path)
    storage.create_job("j2", "a.mp4", "2026-09-30T00:00:00+00:00")
    job_dir = settings.uploads_dir / "j2"
    job_dir.mkdir(parents=True)
    video = job_dir / "a.mp4"
    video.write_bytes(b"v")
    (job_dir / ".options.json").write_text(json.dumps({"quality": "fast"}))
    seen = {}

    def fake_transcribe(path, cfg, logger, out_path):
        seen["model"] = cfg.model
        out_path.write_text("t")
        return out_path

    mgr = JobManager(
        storage, settings,
        cfg_loader=lambda: Config(openai_api_key="k", model="whisper-large-v3", provider="local"),
        transcribe_fn=fake_transcribe,
    )
    mgr._run("j2", video)
    assert seen["model"] == "whisper-large-v3-turbo"
    assert storage.get_job("j2").status == "done"
