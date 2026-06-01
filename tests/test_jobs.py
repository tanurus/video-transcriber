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
