from pathlib import Path

from web.jobs import JobManager
from web.storage import Storage


def _make_video(settings, job_id="j1", name="clip.mp4"):
    d = settings.uploads_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"x")
    return p


def test_run_success(settings):
    store = Storage(settings.db_path)
    store.create_job("j1", "clip.mp4", "2026-06-01T00:00:00+00:00")
    video = _make_video(settings)

    def fake_transcribe(video_path, cfg, logger=None, out_path=None):
        logger("hello")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("transcript text", encoding="utf-8")
        return Path(out_path)

    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=fake_transcribe)
    mgr._run("j1", video)

    job = store.get_job("j1")
    assert job.status == "done"
    assert job.transcript_path == "transcripts/j1.txt"
    assert (settings.transcripts_dir / "j1.txt").read_text(encoding="utf-8") == "transcript text"
    lines, nxt = mgr.get_logs("j1", since=0)
    assert "hello" in lines
    assert nxt == len(lines)


def test_run_error(settings):
    store = Storage(settings.db_path)
    store.create_job("j2", "bad.mp4", "2026-06-01T00:00:00+00:00")
    video = _make_video(settings, job_id="j2", name="bad.mp4")

    def boom(video_path, cfg, logger=None, out_path=None):
        raise RuntimeError("ffmpeg exploded")

    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=boom)
    mgr._run("j2", video)

    job = store.get_job("j2")
    assert job.status == "error"
    assert "ffmpeg exploded" in job.error


class FlakyStorage:
    """Delegates to a real Storage but fails the first update_status call."""

    def __init__(self, real):
        self._real = real
        self.failures_left = 1

    def update_status(self, *args, **kwargs):
        if self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("database is locked")
        return self._real.update_status(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class DeadStorage:
    """Every status write fails — simulates a full disk / dead database."""

    def update_status(self, *args, **kwargs):
        raise RuntimeError("disk full")


def test_run_survives_transient_storage_failure(settings):
    real = Storage(settings.db_path)
    real.create_job("j5", "clip.mp4", "2026-06-01T00:00:00+00:00")
    video = _make_video(settings, job_id="j5")
    store = FlakyStorage(real)

    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=lambda *a, **k: None)
    mgr._run("j5", video)  # must not raise into the (discarded) Future

    job = real.get_job("j5")
    assert job.status == "error"
    assert "locked" in job.error


def test_run_never_raises_even_when_storage_is_dead(settings):
    mgr = JobManager(DeadStorage(), settings, cfg_loader=lambda: None, transcribe_fn=lambda *a, **k: None)
    mgr._run("j6", settings.uploads_dir / "j6" / "x.mp4")  # no exception may escape

    lines, _ = mgr.get_logs("j6")
    assert any(line.startswith("Error:") for line in lines)


def test_final_log_line_lands_before_terminal_status(settings):
    """A poll that sees a terminal status must already see the final log line."""
    real = Storage(settings.db_path)
    real.create_job("j7", "clip.mp4", "2026-06-01T00:00:00+00:00")
    video = _make_video(settings, job_id="j7")

    captured = {}

    class SnoopingStorage:
        def __init__(self, real_storage):
            self._real = real_storage

        def update_status(self, job_id, status, **kwargs):
            if status in ("done", "error"):
                captured["logs_at_terminal_write"] = mgr.get_logs(job_id)[0]
            return self._real.update_status(job_id, status, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def fake_transcribe(video_path, cfg, logger=None, out_path=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("t", encoding="utf-8")
        return Path(out_path)

    mgr = JobManager(
        SnoopingStorage(real), settings, cfg_loader=lambda: None, transcribe_fn=fake_transcribe
    )
    mgr._run("j7", video)

    assert "Done." in captured["logs_at_terminal_write"]


def test_get_logs_since(settings):
    store = Storage(settings.db_path)
    mgr = JobManager(store, settings, cfg_loader=lambda: None, transcribe_fn=lambda *a, **k: None)
    mgr._append_log("x", "a")
    mgr._append_log("x", "b")
    mgr._append_log("x", "c")
    lines, nxt = mgr.get_logs("x", since=1)
    assert lines == ["b", "c"]
    assert nxt == 3
    empty, nxt2 = mgr.get_logs("x", since=3)
    assert empty == []
    assert nxt2 == 3
