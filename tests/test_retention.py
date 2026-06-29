import os

from web.jobs import purge_old_videos


def test_purge_removes_only_old_dirs(settings):
    old = settings.uploads_dir / "old"
    old.mkdir(parents=True)
    (old / "v.mp4").write_bytes(b"x")
    new = settings.uploads_dir / "new"
    new.mkdir(parents=True)
    (new / "v.mp4").write_bytes(b"x")

    now = 1_000_000_000
    old_mtime = now - 40 * 86400  # 40 days old, beyond 30-day retention
    os.utime(old, (old_mtime, old_mtime))
    new_mtime = now - 5 * 86400
    os.utime(new, (new_mtime, new_mtime))

    removed = purge_old_videos(settings, now=now)

    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_purge_no_uploads_dir(tmp_path, settings):
    import shutil
    shutil.rmtree(settings.uploads_dir)
    assert purge_old_videos(settings, now=1_000_000_000) == 0


def _old_dir(settings, name, now):
    d = settings.uploads_dir / name
    d.mkdir(parents=True)
    (d / "v.mp4").write_bytes(b"x")
    mtime = now - 40 * 86400
    os.utime(d, (mtime, mtime))
    return d


def test_purge_skips_dirs_of_active_jobs(settings):
    from web.storage import Storage

    now = 1_000_000_000
    queued_dir = _old_dir(settings, "queued-job", now)
    done_dir = _old_dir(settings, "done-job", now)

    store = Storage(settings.db_path)
    store.create_job("queued-job", "a.mp4", "2026-06-01T00:00:00+00:00")
    store.create_job("done-job", "b.mp4", "2026-06-01T00:00:00+00:00")
    store.update_status("done-job", "done")

    removed = purge_old_videos(settings, now=now, storage=store)

    assert removed == 1
    assert queued_dir.exists(), "videos of jobs still waiting to run must never be purged"
    assert not done_dir.exists()


def test_purge_continues_past_bad_entries(settings):
    now = 1_000_000_000
    bad_dir = _old_dir(settings, "aaa-bad", now)
    good_dir = _old_dir(settings, "zzz-good", now)

    class ExplodingStorage:
        def get_job(self, job_id):
            if job_id == "aaa-bad":
                raise RuntimeError("db hiccup")
            return None

    removed = purge_old_videos(settings, now=now, storage=ExplodingStorage())

    assert good_dir.exists() is False, "one bad entry must not abort the whole sweep"
    assert bad_dir.exists(), "the entry that errored is skipped, not deleted"
    assert removed == 1
