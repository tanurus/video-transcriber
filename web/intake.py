"""One door for every recording, whatever sent it (browser, phone, API, desktop app).

Hashes the file (so the same recording sent from two devices is recognised),
works out the best recording time, labels the source device, resolves the
settings and creates the job.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from werkzeug.utils import secure_filename

from app import options as O

from .storage import Job, Storage, utcnow_iso

_FILENAME_DATES = [
    # 2026-09-29 10-15-00, 2026-09-29_10.15.00, 2026-09-29T10:15
    re.compile(r"(20\d\d)[-_.](\d\d)[-_.](\d\d)[ _T-]+(\d\d)[-_.:h](\d\d)(?:[-_.:m](\d\d))?"),
    # 20260929_101500, VID_20260929_101500, Recording 20260929101500
    re.compile(r"(20\d\d)(\d\d)(\d\d)[_T-]?(\d\d)(\d\d)(\d\d)"),
    # date only: 2026-09-29
    re.compile(r"(20\d\d)-(\d\d)-(\d\d)"),
]


def _local_zone():
    name = os.getenv("TZ", "").lstrip(":")
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001 - unknown zone name: fall back to the system zone
            pass
    return datetime.now().astimezone().tzinfo


def recorded_at_from_name(name: str) -> Optional[str]:
    for rx in _FILENAME_DATES:
        m = rx.search(name)
        if not m:
            continue
        parts = [int(g) if g else 0 for g in m.groups()] + [0, 0, 0]
        try:
            dt = datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
        except ValueError:
            continue
        # Recorder file names carry local wall-clock time; stamp it with the
        # recorder's zone (TZ, i.e. this server's) so Supabase stores the true instant.
        return dt.replace(tzinfo=_local_zone()).isoformat()
    return None


def recorded_at_from_epoch_ms(value: Any) -> Optional[str]:
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def device_label(user_agent: str) -> str:
    ua = user_agent or ""
    if "iPhone" in ua:
        device = "iPhone"
    elif "iPad" in ua:
        device = "iPad"
    elif "Android" in ua:
        m = re.search(r"Android [\d.]+; ([^;)]+?)(?: Build|\))", ua)
        device = f"Android ({m.group(1).strip()})" if m else "Android"
    elif "Windows" in ua:
        device = "Windows"
    elif "Macintosh" in ua:
        device = "Mac"
    elif "Linux" in ua:
        device = "Linux"
    else:
        device = "unknown device"
    for name, token in (("Edge", "Edg/"), ("Chrome", "Chrome/"), ("Firefox", "Firefox/"), ("Safari", "Safari/")):
        if token in ua:
            return f"{device} · {name}"
    return device if ua else ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(filename: str, ext: str) -> str:
    name = secure_filename(filename)
    # secure_filename strips non-ASCII: "Видео.mp4" becomes "mp4" (no dot).
    # ffmpeg sniffs by content but keep an extension so tools recognise the file.
    if not name or "." not in name:
        name = f"upload.{ext}" if ext else "upload.bin"
    return name


@dataclass
class IntakeResult:
    job_id: str
    duplicate: bool = False
    duplicate_of: Optional[Job] = None


def resolve_job_options(storage: Storage, raw: Optional[Dict[str, Any]], preset: Optional[str]) -> Dict[str, Any]:
    """User defaults (Settings) < preset < this request, then the chosen profile."""
    base = storage.get_setting("default_options", {}) or {}
    opts = O.resolve(raw or {}, base=base, preset=preset)
    profile_id = str(opts.get("profile_id") or "")
    if profile_id.isdigit():
        opts = O.apply_profile(opts, storage.get_profile(int(profile_id)))
    opts["_summary"] = O.summary(opts)
    return opts


def ingest(
    storage: Storage,
    uploads_dir: Path,
    temp_file: Path,
    original_name: str,
    *,
    raw_options: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
    source: str = "web",
    source_detail: str = "",
    recorded_at: Optional[str] = None,
    title: Optional[str] = None,
    on_duplicate: str = "skip",
    sha256: Optional[str] = None,
) -> IntakeResult:
    """Move ``temp_file`` into the library and create its job (not yet submitted).

    With ``on_duplicate="skip"`` a file whose exact bytes are already in the
    library is not transcribed again; the existing job is returned instead.
    """
    digest = sha256 or sha256_file(temp_file)
    if on_duplicate != "new":
        existing = [j for j in storage.find_by_sha(digest) if j.status in ("queued", "running", "done")]
        if existing:
            temp_file.unlink(missing_ok=True)
            return IntakeResult(job_id=existing[0].id, duplicate=True, duplicate_of=existing[0])

    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    job_id = uuid.uuid4().hex
    dest_dir = uploads_dir / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name(original_name, ext)
    shutil.move(str(temp_file), dest)

    opts = resolve_job_options(storage, raw_options, preset)
    storage.create_job(
        job_id, original_name, utcnow_iso(),
        title=(title or "").strip()[:120] or None,
        source=source, source_detail=source_detail[:200] or None,
        recorded_at=recorded_at or recorded_at_from_name(original_name),
        file_size=dest.stat().st_size, file_sha256=digest,
        settings_json=json.dumps(opts),
    )
    return IntakeResult(job_id=job_id)


# --- resumable chunked uploads (browser) ------------------------------------------------

class UploadError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ChunkedUploads:
    """Upload a big file in pieces; any piece can be retried and a dropped
    connection resumes from what the server already has."""

    CHUNK_SIZE = 8 * 1024 * 1024

    def __init__(self, root: Path, allowed_ext: frozenset, max_bytes: int) -> None:
        self.root = Path(root)
        self.allowed_ext = allowed_ext
        self.max_bytes = max_bytes

    def _dir(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id or ""):
            raise UploadError("Unknown upload.", 404)
        d = self.root / upload_id
        if not d.is_dir():
            raise UploadError("Unknown or expired upload.", 404)
        return d

    def _meta(self, upload_id: str) -> Dict[str, Any]:
        return json.loads((self._dir(upload_id) / "meta.json").read_text(encoding="utf-8"))

    def start(self, filename: str, size: int, last_modified: Any = None) -> Dict[str, Any]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in self.allowed_ext:
            raise UploadError(f"Unsupported file type: .{ext}" if ext else "The file has no extension.")
        if size <= 0:
            raise UploadError("The file is empty.")
        if size > self.max_bytes:
            raise UploadError(f"File too large ({size / 1e9:.1f} GB); the limit is {self.max_bytes / 1e9:.1f} GB.", 413)
        upload_id = uuid.uuid4().hex
        d = self.root / upload_id
        d.mkdir(parents=True)
        (d / "data.part").touch()
        (d / "meta.json").write_text(json.dumps({
            "filename": filename, "size": size, "last_modified": last_modified, "created_at": utcnow_iso(),
        }), encoding="utf-8")
        return {"upload_id": upload_id, "chunk_size": self.CHUNK_SIZE, "received": 0}

    def status(self, upload_id: str) -> Dict[str, Any]:
        meta = self._meta(upload_id)
        received = (self._dir(upload_id) / "data.part").stat().st_size
        return {"upload_id": upload_id, "received": received, "size": meta["size"]}

    def append(self, upload_id: str, offset: int, data: bytes) -> Dict[str, Any]:
        d = self._dir(upload_id)
        meta = self._meta(upload_id)
        part = d / "data.part"
        have = part.stat().st_size
        if offset + len(data) <= have:
            return {"received": have}  # a retried chunk we already stored
        if offset != have:
            raise UploadError(f"Out of order chunk: server has {have} bytes.", 409)
        if have + len(data) > meta["size"]:
            raise UploadError("More data than the declared file size.", 400)
        with open(part, "ab") as f:
            f.write(data)
        return {"received": have + len(data)}

    def finish(self, upload_id: str) -> tuple:
        """Return (path, meta) once every byte has arrived."""
        d = self._dir(upload_id)
        meta = self._meta(upload_id)
        part = d / "data.part"
        if part.stat().st_size != meta["size"]:
            raise UploadError(f"Upload incomplete: {part.stat().st_size} of {meta['size']} bytes.", 409)
        return part, meta

    def discard(self, upload_id: str) -> None:
        try:
            shutil.rmtree(self._dir(upload_id), ignore_errors=True)
        except UploadError:
            pass


def duplicates_summary(jobs: List[Job]) -> List[Dict[str, str]]:
    return [{"id": j.id, "name": j.display_name, "created_at": j.created_at} for j in jobs]
