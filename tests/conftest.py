import json

import pytest

from web.settings import WebSettings


@pytest.fixture
def settings(tmp_path):
    s = WebSettings(
        data_dir=tmp_path,
        retain_video_days=30,
        max_content_mb=2048,
        allowed_ext=frozenset({"mp4", "mkv", "mov", "avi", "webm", "m4a", "mp3", "wav"}),
        max_file_mb=64,
    )
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.transcripts_dir.mkdir(parents=True, exist_ok=True)
    return s


class Result:
    def __init__(self, text="the transcript", segments=None, languages=("en",)):
        self.text = text
        self.segments = segments or [{"start": 0.0, "end": 2.0, "text": text}]
        self.languages = list(languages)
        self.duration = 2.0
        self.provider = "local"
        self.model = "whisper-large-v3"


@pytest.fixture
def make_app(settings):
    """Real app + real JobManager; the GPU queue only records submissions."""
    from web import create_app
    from web.connections import ConnectionStore
    from web.jobs import JobManager
    from web.storage import Storage

    def build(ai_factory=None, sync_fn=None, transcribe_fn=None):
        storage = Storage(settings.db_path)
        conns = ConnectionStore(settings.data_dir / "connections.json")

        class RecordingJobs(JobManager):
            submitted = []

            def submit(self, job_id, video_path=None):
                self.submitted.append(job_id)

        RecordingJobs.submitted = []

        def fake_transcribe(source, cfg, opts, kept, logger=None, rnnoise_model=None):
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_bytes(b"flac")
            logger("transcribing")
            return Result()

        jobs = RecordingJobs(storage, settings, cfg_loader=lambda: object(),
                             transcribe_fn=transcribe_fn or fake_transcribe, connections=conns,
                             ai_factory=ai_factory, sync_fn=sync_fn, backend_wait=0)
        app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False,
                         connections=conns, ai_factory=ai_factory)
        app.config.update(TESTING=True)
        client = app.test_client()
        client.storage, client.jobs, client.conns, client.app = storage, jobs, conns, app
        return client

    return build


@pytest.fixture
def client(make_app):
    return make_app()


@pytest.fixture
def done_job(client, settings):
    """A finished job with raw text, segments and kept audio."""
    storage, jobs = client.storage, client.jobs
    storage.create_job("j1", "Meeting.mp4", "2026-09-30T10:00:00+00:00",
                       settings_json=json.dumps({"_summary": "large-v3"}), file_sha256="abc")
    up = settings.uploads_dir / "j1"
    up.mkdir(parents=True)
    (up / "Meeting.mp4").write_bytes(b"video")
    jobs._run("j1")
    return storage.get_job("j1")
