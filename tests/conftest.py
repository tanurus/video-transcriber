import pytest

from web.settings import WebSettings


@pytest.fixture
def settings(tmp_path):
    s = WebSettings(
        data_dir=tmp_path,
        retain_video_days=30,
        max_content_mb=2048,
        allowed_ext=frozenset({"mp4", "mkv", "mov", "avi", "webm", "m4a", "mp3", "wav"}),
    )
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.transcripts_dir.mkdir(parents=True, exist_ok=True)
    return s
