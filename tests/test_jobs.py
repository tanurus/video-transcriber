from web.jobs import JobManager
from web.storage import Storage

from tests.conftest import Result


def _mgr(store, settings, fn):
    return JobManager(store, settings, cfg_loader=lambda: object(), transcribe_fn=fn, backend_wait=0)


def _prep(settings, store, job_id):
    store.create_job(job_id, "clip.mp4", "2026-06-01T00:00:00+00:00")
    d = settings.uploads_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "clip.mp4").write_bytes(b"x")


def _ok(source, cfg, opts, kept, logger=None, rnnoise_model=None):
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"a")
    return Result("t")


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
    def update_status(self, *args, **kwargs):
        raise RuntimeError("disk full")


def test_run_survives_transient_storage_failure(settings):
    real = Storage(settings.db_path)
    _prep(settings, real, "j5")
    _mgr(FlakyStorage(real), settings, _ok)._run("j5")  # must not raise into the (discarded) Future
    job = real.get_job("j5")
    assert job.status == "error" and "locked" in job.error


def test_run_never_raises_even_when_storage_is_dead(settings):
    mgr = _mgr(DeadStorage(), settings, _ok)
    mgr._run("j6")
    assert any(line.startswith("Error:") for line in mgr.get_logs("j6")[0])


def test_final_log_line_lands_before_terminal_status(settings):
    """A poll that sees a terminal status must already see the final log line."""
    real = Storage(settings.db_path)
    _prep(settings, real, "j7")
    captured = {}

    class Snooping:
        def __init__(self, r):
            self._real = r

        def update_status(self, job_id, status, **kwargs):
            if status in ("done", "error"):
                captured["logs"] = mgr.get_logs(job_id)[0]
            return self._real.update_status(job_id, status, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    mgr = _mgr(Snooping(real), settings, _ok)
    mgr._run("j7")
    assert "Done." in captured["logs"]


def test_get_logs_since(settings):
    mgr = _mgr(Storage(settings.db_path), settings, _ok)
    for line in "abc":
        mgr._append_log("x", line)
    assert mgr.get_logs("x", since=1) == (["b", "c"], 3)
    assert mgr.get_logs("x", since=3) == ([], 3)


def test_resume_requeues_queued_and_interrupts_orphans(settings):
    store = Storage(settings.db_path)
    _prep(settings, store, "q1")
    store.create_job("q2", "gone.mp4", "2026-06-01T00:00:00+00:00")  # no upload, no audio
    mgr = _mgr(store, settings, _ok)
    seen = []
    mgr.submit = lambda job_id, video_path=None: seen.append(job_id)
    mgr.resume()
    assert seen == ["q1"]
    assert store.get_job("q2").status == "interrupted"
