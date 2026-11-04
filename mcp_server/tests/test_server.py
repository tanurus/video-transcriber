"""The MCP layer: tools are registered, and errors come back as text instead of crashing."""
import json

import pytest

pytest.importorskip("mcp")  # the web image runs the main suite without the MCP SDK

import anyio  # noqa: E402

from transcriber_mcp import server  # noqa: E402
from transcriber_mcp.client import TranscriberError  # noqa: E402

EXPECTED = {
    "transcriber_status", "get_settings_catalog", "get_default_settings", "set_default_settings",
    "list_profiles", "save_profile", "delete_profile", "transcribe_file", "wait_for_job",
    "search_transcripts", "get_transcript", "list_versions", "regenerate_transcripts",
    "run_ai_finishing", "send_to_supabase",
}


def test_all_tools_are_registered_with_descriptions():
    tools = anyio.run(server.mcp.list_tools)
    assert {t.name for t in tools} == EXPECTED
    assert all(t.description for t in tools)
    no_delete = {t.name for t in tools} - {"delete_profile"}
    assert not any("delete" in n for n in no_delete)  # recordings cannot be deleted over MCP


class Fake:
    base_url = "http://transcriber"

    def get_defaults(self):
        return {"options": {"model": "whisper-large-v3"}}

    def catalog(self):
        raise TranscriberError("401: Invalid or revoked API token.")


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(server, "_client", lambda: Fake())


def test_tool_returns_json(fake):
    out = anyio.run(server.get_default_settings)
    assert json.loads(out)["options"]["model"] == "whisper-large-v3"


def test_tool_turns_server_errors_into_readable_text(fake):
    assert anyio.run(server.get_settings_catalog) == "Error: 401: Invalid or revoked API token."


def test_missing_configuration_is_explained(monkeypatch):
    monkeypatch.delenv("TRANSCRIBER_URL", raising=False)
    with pytest.raises(TranscriberError, match="TRANSCRIBER_URL"):
        server.TranscriberClient.from_env()
