from web import create_app
from web.storage import Storage


class FakeJobs:
    def __init__(self):
        self.submitted = []

    def submit(self, job_id, video_path):
        self.submitted.append((job_id, str(video_path)))

    def get_logs(self, job_id, since=0):
        return [], 0


def _boot(settings, storage):
    jobs = FakeJobs()
    app = create_app(settings=settings, storage=storage, job_manager=jobs, start_background=False)
    app.config.update(TESTING=True)
    return jobs


def test_queued_jobs_requeued_on_boot(settings):
    storage = Storage(settings.db_path)
    storage.create_job("q1", "clip.mp4", "2026-06-01T00:00:00+00:00")
    video_dir = settings.uploads_dir / "q1"
    video_dir.mkdir(parents=True)
    video = video_dir / "clip.mp4"
    video.write_bytes(b"x")

    jobs = _boot(settings, storage)

    assert ("q1", str(video)) in jobs.submitted
    assert storage.get_job("q1").status == "queued"


def test_queued_job_with_missing_video_marked_interrupted(settings):
    storage = Storage(settings.db_path)
    storage.create_job("gone", "clip.mp4", "2026-06-01T00:00:00+00:00")
    # no upload directory on disk

    jobs = _boot(settings, storage)

    assert jobs.submitted == []
    assert storage.get_job("gone").status == "interrupted"


def test_running_job_marked_interrupted_on_boot(settings):
    storage = Storage(settings.db_path)
    storage.create_job("r1", "clip.mp4", "2026-06-01T00:00:00+00:00")
    storage.update_status("r1", "running")

    jobs = _boot(settings, storage)

    assert jobs.submitted == []
    assert storage.get_job("r1").status == "interrupted"
