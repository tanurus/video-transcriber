"""The MCP server's HTTP client, end to end against the real app (no network, no mocks of our code)."""
import hashlib
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_server"))

from transcriber_mcp.client import TranscriberClient, TranscriberError  # noqa: E402

TOKEN = "vt_client-test"


class FlakyOnce(httpx.WSGITransport):
    """Fails the first chunk PUT after the server stored it, like a dropped connection."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.failed = False

    def handle_request(self, request):
        resp = super().handle_request(request)
        if request.method == "PUT" and "/uploads/" in request.url.path and not self.failed:
            self.failed = True
            raise httpx.ReadError("connection reset", request=request)
        return resp


@pytest.fixture
def api(client):
    client.storage.add_token("laptop-mcp", hashlib.sha256(TOKEN.encode()).hexdigest(), TOKEN[:10])
    transport = FlakyOnce(app=client.app)
    c = TranscriberClient("http://transcriber", TOKEN, transport=transport)
    c.flask, c.transport = client, transport
    return c


def test_settings_round_trip(api):
    assert any(o["key"] == "beam_size" for o in api.catalog()["options"])
    api.set_defaults({"language_mode": "fixed", "language": "ro"}, preset="balanced")
    got = api.get_defaults()["options"]
    assert got["language"] == "ro" and got["preset"] == "balanced"
    p = api.save_profile(name="Sales", hotwords="BusinessClass", languages="en,ro")
    assert api.save_profile(id=p["id"], notes="weekly")["hotwords"] == "BusinessClass"
    assert [x["name"] for x in api.list_profiles()] == ["Sales"]
    api.delete_profile(p["id"])
    assert api.list_profiles() == []
    assert api.connections()["ai_ready"] is False
    assert "counts" in api.health()


def test_upload_resumes_after_a_dropped_chunk_and_creates_a_job(api, tmp_path, monkeypatch):
    monkeypatch.setattr("transcriber_mcp.client.time.sleep", lambda s: None)
    rec = tmp_path / "2026-11-04 call.m4a"
    rec.write_bytes(b"a" * 5000)
    res = api.upload(str(rec), preset="fast", title="Call", language="ro",
                     options={"language_mode": "fixed"})
    assert api.transport.failed  # the drop happened and the client recovered
    job = res["job"]
    assert res["duplicate"] is False and job["status"] == "queued" and job["source"] == "mcp"
    assert job["source_detail"] == "laptop-mcp"
    assert job["id"] in api.flask.jobs.submitted
    stored = api.get_job(job["id"])
    assert stored["settings"]["preset"] == "fast" and stored["settings"]["language"] == "ro"
    # Same bytes again: recognised, not transcribed twice.
    again = api.upload(str(rec))
    assert again["duplicate"] is True
    assert [j["id"] for j in api.list_jobs(status="queued")] == [job["id"]]


def test_library_and_actions(api, done_job):
    assert [j["id"] for j in api.list_jobs(query="Meeting")] == ["j1"]
    full = api.get_job("j1", text=True)
    assert full["transcript_raw"].strip() == "the transcript"
    assert api.segments("j1")[0]["text"] == "the transcript"
    created = api.action("regenerate", ["j1"], preset="fast")["created"]
    assert len(api.versions("j1")) == 2 and created
    with pytest.raises(TranscriberError, match="OpenAI"):
        api.action("ai", ["j1"])


def test_wait_returns_when_done_or_on_timeout(api, done_job):
    assert api.wait("j1", timeout_sec=1, sleep=lambda s: None)["status"] == "done"
    api.flask.storage.update_status("j1", "running")
    assert api.wait("j1", timeout_sec=0, sleep=lambda s: None)["status"] == "running"


def test_errors_are_readable(api, tmp_path):
    bad = TranscriberClient("http://transcriber", "vt_wrong", transport=httpx.WSGITransport(app=api.flask.app))
    with pytest.raises(TranscriberError, match="401: Invalid or revoked"):
        bad.health()
    with pytest.raises(TranscriberError, match="No such file"):
        api.upload(str(tmp_path / "missing.mp4"))
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(TranscriberError, match="Unsupported file type"):
        api.upload(str(txt))
