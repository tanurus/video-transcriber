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
