"""HTTP client for the transcriber's token API (/api/v1). Only depends on httpx."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

DONE_STATES = {"done", "error", "interrupted", "cancelled"}


class TranscriberError(RuntimeError):
    """The server answered with an error (its message is kept) or could not be reached."""


class TranscriberClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        if not base_url:
            raise TranscriberError("TRANSCRIBER_URL is not set (e.g. http://<machine>.<tailnet>.ts.net:18920).")
        if not token:
            raise TranscriberError("TRANSCRIBER_TOKEN is not set (create one in the web app: Settings → API tokens).")
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url + "/api/v1", timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {token}",
                                           "User-Agent": "transcriber-mcp"})

    @classmethod
    def from_env(cls) -> "TranscriberClient":
        return cls(os.getenv("TRANSCRIBER_URL", ""), os.getenv("TRANSCRIBER_TOKEN", ""))

    def close(self) -> None:
        self._http.close()

    # --- plumbing ------------------------------------------------------------------------------

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise TranscriberError(f"Cannot reach {self.base_url}: {e}. Is this PC on the tailnet?") from e
        try:
            data = r.json()
        except ValueError:
            data = None
        if r.status_code >= 400:
            msg = data.get("error") if isinstance(data, dict) else r.text[:300]
            raise TranscriberError(f"{r.status_code}: {msg}")
        return data

    # --- settings ------------------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._req("GET", "/health")

    def connections(self) -> Dict[str, Any]:
        return self._req("GET", "/connections")

    def catalog(self) -> Dict[str, Any]:
        return self._req("GET", "/settings/catalog")

    def get_defaults(self) -> Dict[str, Any]:
        return self._req("GET", "/settings/defaults")

    def set_defaults(self, options: Optional[Dict[str, Any]] = None, preset: Optional[str] = None,
                     merge: bool = True) -> Dict[str, Any]:
        return self._req("PUT", "/settings/defaults",
                         json={"options": options or {}, "preset": preset, "merge": merge})

    def list_profiles(self) -> List[Dict[str, Any]]:
        return self._req("GET", "/profiles")["profiles"]

    def save_profile(self, **fields: Any) -> Dict[str, Any]:
        body = {k: v for k, v in fields.items() if v is not None}
        return self._req("POST", "/profiles", json=body)["profile"]

    def delete_profile(self, profile_id: int) -> Dict[str, Any]:
        return self._req("DELETE", f"/profiles/{int(profile_id)}")

    # --- library -------------------------------------------------------------------------------

    def list_jobs(self, query: str = "", status: str = "", latest_only: bool = True,
                  limit: int = 20) -> List[Dict[str, Any]]:
        params = {"q": query, "status": status, "latest_only": "1" if latest_only else "0", "limit": limit}
        return self._req("GET", "/jobs", params={k: v for k, v in params.items() if v not in ("", None)})["jobs"]

    def get_job(self, job_id: str, text: bool = False) -> Dict[str, Any]:
        return self._req("GET", f"/jobs/{job_id}", params={"text": "1"} if text else None)

    def segments(self, job_id: str) -> List[Dict[str, Any]]:
        return self._req("GET", f"/jobs/{job_id}/segments")["segments"]

    def versions(self, job_id: str) -> List[Dict[str, Any]]:
        return self._req("GET", f"/jobs/{job_id}/versions")["versions"]

    def action(self, action: str, ids: Iterable[str], **extra: Any) -> Dict[str, Any]:
        body = {"action": action, "ids": list(ids)}
        body.update({k: v for k, v in extra.items() if v is not None})
        return self._req("POST", "/jobs/actions", json=body)

    def wait(self, job_id: str, timeout_sec: float = 1800, poll_sec: float = 5.0,
             sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
        """Poll until the job finishes or the timeout passes; returns the last state either way."""
        deadline = time.monotonic() + timeout_sec
        while True:
            job = self.get_job(job_id)
            if job["status"] in DONE_STATES or time.monotonic() >= deadline:
                return job
            sleep(poll_sec)

    # --- upload --------------------------------------------------------------------------------

    def upload(self, path: str, *, progress: Optional[Callable[[int, int], None]] = None,
               retries: int = 5, **job_fields: Any) -> Dict[str, Any]:
        """Send a local file in resumable pieces, then create the job.

        ``job_fields``: preset, options, profile_id, language, prompt, hotwords, model,
        title, recorded_at, on_duplicate, source, source_detail.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise TranscriberError(f"No such file on this PC: {p}")
        size = p.stat().st_size
        start = self._req("POST", "/uploads", json={
            "filename": p.name, "size": size, "last_modified": int(p.stat().st_mtime * 1000)})
        upload_id, chunk = start["upload_id"], int(start["chunk_size"])
        offset = 0
        failures = 0
        with p.open("rb") as f:
            while offset < size:
                f.seek(offset)
                piece = f.read(chunk)
                try:
                    offset = self._req("PUT", f"/uploads/{upload_id}", params={"offset": offset},
                                       content=piece, timeout=300)["received"]
                    failures = 0
                except TranscriberError:
                    failures += 1
                    if failures > retries:
                        raise
                    time.sleep(min(30, 2 ** failures))
                    # Resume from whatever the server actually has.
                    offset = self._req("GET", f"/uploads/{upload_id}")["received"]
                if progress:
                    progress(offset, size)
        body = {k: v for k, v in job_fields.items() if v is not None}
        body.setdefault("source", "mcp")
        return self._req("POST", f"/uploads/{upload_id}/complete", json=body, timeout=600)
