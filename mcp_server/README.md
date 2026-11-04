# transcriber-mcp

An MCP server for the video-transcriber web app. Run it on the computer you work on
(Claude Code, Claude Desktop, or any MCP client) and it talks to the transcriber over
the token API, for example across a Tailscale network.

## What it can do

| Area | Tools |
|---|---|
| Transcribe | `transcribe_file` (uploads a file from this computer, any size, resumable; `wait=true` returns the text), `wait_for_job` |
| Library | `search_transcripts`, `get_transcript` (raw / AI-cleaned, timestamps, paging), `list_versions`, `regenerate_transcripts`, `run_ai_finishing`, `send_to_supabase` |
| Configuration | `get_settings_catalog`, `get_default_settings`, `set_default_settings`, `list_profiles`, `save_profile`, `delete_profile`, `transcriber_status` |

It deliberately **cannot** change the OpenAI / Supabase keys, create API tokens or delete
recordings. Those stay in the web app.

## Setup

1. In the web app open **Settings → API tokens**, create a token (e.g. `laptop-mcp`) and copy it.
2. Make sure this computer can open the web app (same tailnet).
3. Add the server. It needs [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv` or
   `pip install uv`).

**Claude Code**

```
claude mcp add transcriber --scope user \
  -e TRANSCRIBER_URL=http://<machine>.<tailnet>.ts.net:18920 \
  -e TRANSCRIBER_TOKEN=vt_... \
  -- uvx --from "git+https://github.com/tanurus/video-transcriber#subdirectory=mcp_server" transcriber-mcp
```

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/tanurus/video-transcriber#subdirectory=mcp_server", "transcriber-mcp"],
      "env": {
        "TRANSCRIBER_URL": "http://<machine>.<tailnet>.ts.net:18920",
        "TRANSCRIBER_TOKEN": "vt_..."
      }
    }
  }
}
```

To pick up a newer version later: `uv cache clean transcriber-mcp` (or add `--refresh` to the
`uvx` arguments once).

## Try it

- "What are the transcriber's current default settings?"
- "Make Romanian the fixed default language and use the balanced preset."
- "Create a profile *Kayak QBR* with hotwords Kayak, BWK, eCPC and languages en,ro."
- "Transcribe `C:\Users\me\Videos\meeting.mkv` with that profile and give me the text."
- "Regenerate last week's sales call with the fast preset and compare the versions."

## API it uses

All under `/api/v1` with `Authorization: Bearer vt_...`: `settings/catalog`, `settings/defaults`
(GET/PUT), `profiles` (GET/POST/DELETE), `connections`, `health`, `uploads` (resumable),
`jobs`, `jobs/<id>`, `jobs/<id>/segments`, `jobs/<id>/versions`, `jobs/actions`.

## Development

```
cd mcp_server
uv run --with pytest pytest -q tests
```

The HTTP client is also tested end-to-end against the real Flask app in the main suite
(`tests/test_mcp_client.py`).
