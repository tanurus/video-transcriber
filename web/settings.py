from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOWED_EXT = "mp4,mkv,mov,avi,webm,m4a,mp3,wav,ogg,opus,flac,aac,amr,3gp,wma,mpeg,mpg,m4v,ts"


@dataclass(frozen=True)
class WebSettings:
    data_dir: Path
    retain_video_days: int
    max_content_mb: int
    allowed_ext: frozenset[str]
    # Largest single recording accepted through chunked / API upload.
    max_file_mb: int = 8192

    @staticmethod
    def load() -> "WebSettings":
        # "or default" so blank env values ("RETAIN_VIDEO_DAYS=") behave like unset.
        raw_ext = os.getenv("ALLOWED_EXT") or DEFAULT_ALLOWED_EXT
        allowed = frozenset(
            e.strip().lower().lstrip(".") for e in raw_ext.split(",") if e.strip()
        )
        return WebSettings(
            data_dir=Path(os.getenv("DATA_DIR") or "/data"),
            retain_video_days=int(os.getenv("RETAIN_VIDEO_DAYS") or "30"),
            max_content_mb=int(os.getenv("MAX_CONTENT_MB") or "2048"),
            allowed_ext=allowed,
            max_file_mb=int(os.getenv("MAX_FILE_MB") or "8192"),
        )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"
