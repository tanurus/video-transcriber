"""MCP server (stdio) for the video-transcriber web app.

Runs on the machine you work on, talks to the transcriber over its token API:

    TRANSCRIBER_URL=http://<machine>.<tailnet>.ts.net:18920
    TRANSCRIBER_TOKEN=vt_...            # web app → Settings → API tokens

It can upload local recordings, read and search transcripts, regenerate them with
other settings, and change the transcription defaults and profiles. It cannot
change API keys, mint tokens or delete recordings — those stay in the web UI.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .client import TranscriberClient, TranscriberError

INSTRUCTIONS = """\
Video Transcriber: local GPU Whisper (large-v3) running on Andrew's VPS, reached over Tailscale.
- Before changing settings, call get_settings_catalog to see every option, its range and choices.
- Presets: accuracy (default), balanced, fast. Mixed RO/RU/EN meetings: keep language_mode=auto.
- transcribe_file takes a path on THIS computer and uploads it; set wait=true to get the text back.
- Defaults changed with set_default_settings apply to every device's new uploads.
- Errors come back as text starting with "Error:"; read them, they say what to fix."""

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

mcp = MCPServer("transcriber", instructions=INSTRUCTIONS)


@lru_cache(maxsize=1)
def _client() -> TranscriberClient:
    return TranscriberClient.from_env()


async def _call(fn, *args: Any, **kwargs: Any) -> str:
    """Run a blocking client call off the event loop; return JSON text or a readable error."""
    try:
        result = await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
    except TranscriberError as e:
        return f"Error: {e}"
    return json.dumps(result, ensure_ascii=False, indent=1)


def _job_brief(job: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("id", "name", "title", "status", "error", "ai_status", "sync_status", "created_at",
            "recorded_at", "duration_sec", "languages", "version", "settings_summary", "tags")
    return {k: job.get(k) for k in keys if job.get(k) not in (None, "", [])}


# --- status & settings -------------------------------------------------------------------------

@mcp.tool(annotations=READ)
async def transcriber_status() -> str:
    """GPU server health, library counts, and whether AI finishing / Supabase are connected."""
    def run() -> Dict[str, Any]:
        c = _client()
        return {"server": c.base_url, "health": c.health(), "connections": c.connections()}
    return await _call(run)


@mcp.tool(annotations=READ)
async def get_settings_catalog() -> str:
    """Every transcription option: key, type, default, range/choices, help text, plus presets and
    the profile choices. Read this before set_default_settings or passing options."""
    return await _call(lambda: _client().catalog())


@mcp.tool(annotations=READ)
async def get_default_settings() -> str:
    """The settings every device's new uploads start from, with a one-line summary."""
    return await _call(lambda: _client().get_defaults())


@mcp.tool(annotations=WRITE)
async def set_default_settings(options: Optional[Dict[str, Any]] = None,
                               preset: Optional[Literal["accuracy", "balanced", "fast"]] = None,
                               replace: bool = False) -> str:
    """Change the default transcription settings for all devices.

    options: {option_key: value} from get_settings_catalog, e.g. {"language_mode": "fixed", "language": "ro"}.
    preset: apply a preset's values first. replace=true starts from the catalogue defaults
    instead of keeping the current ones. Out-of-range numbers are clamped; unknown keys are rejected.
    """
    return await _call(lambda: _client().set_defaults(options, preset, merge=not replace))


@mcp.tool(annotations=READ)
async def list_profiles() -> str:
    """Transcription profiles: vocabulary prompt, hotwords and candidate languages for recurring meetings."""
    return await _call(lambda: _client().list_profiles())


@mcp.tool(annotations=WRITE)
async def save_profile(name: Optional[str] = None, prompt: Optional[str] = None, hotwords: Optional[str] = None,
                       languages: Optional[str] = None, notes: Optional[str] = None,
                       profile_id: Optional[int] = None) -> str:
    """Create a profile, or update one by profile_id (fields you leave out keep their value).

    prompt: context sentence with names/jargon. hotwords: comma-separated terms to favour.
    languages: candidate languages such as "ro,ru,en" (switches auto-detect to a per-chunk race).
    """
    return await _call(lambda: _client().save_profile(id=profile_id, name=name, prompt=prompt,
                                                      hotwords=hotwords, languages=languages, notes=notes))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False))
async def delete_profile(profile_id: int) -> str:
    """Delete a transcription profile (transcripts made with it are not affected)."""
    return await _call(lambda: _client().delete_profile(profile_id))


# --- transcribe ----------------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
async def transcribe_file(path: str, preset: Optional[Literal["accuracy", "balanced", "fast"]] = None,
                          language: Optional[str] = None, profile_id: Optional[int] = None,
                          options: Optional[Dict[str, Any]] = None, title: Optional[str] = None,
                          wait: bool = False, timeout_minutes: int = 60) -> str:
    """Upload a recording from THIS computer (video or audio, any size, resumable) and transcribe it.

    language: ISO code to force one language (leave empty for mixed-language meetings).
    options: per-job overrides from get_settings_catalog. The same file uploaded twice is
    recognised and not transcribed again. wait=true blocks until done and returns the text.
    """
    def run() -> Dict[str, Any]:
        c = _client()
        opts = dict(options or {})
        if language:
            opts.setdefault("language_mode", "fixed")
        res = c.upload(path, preset=preset, options=opts, language=language,
                       profile_id=str(profile_id) if profile_id else None, title=title)
        job = res["job"]
        out: Dict[str, Any] = {"duplicate": res["duplicate"], "job": _job_brief(job),
                               "web_url": c.base_url + job.get("url", "")}
        if wait and job["status"] not in ("done", "error"):
            job = c.wait(job["id"], timeout_sec=timeout_minutes * 60)
            out["job"] = _job_brief(job)
        if out["job"]["status"] == "done" and wait:
            full = c.get_job(job["id"], text=True)
            out["transcript"] = full.get("transcript_clean") or full.get("transcript_raw")
        return out
    return await _call(run)


@mcp.tool(annotations=READ)
async def wait_for_job(job_id: str, timeout_minutes: int = 30) -> str:
    """Wait until a transcription finishes (or the timeout passes) and return its status."""
    return await _call(lambda: _job_brief(_client().wait(job_id, timeout_sec=timeout_minutes * 60)))


# --- library -------------------------------------------------------------------------------------

@mcp.tool(annotations=READ)
async def search_transcripts(query: str = "", status: str = "", latest_only: bool = True, limit: int = 20) -> str:
    """Search the library by title, file name, description or tag. status: queued|running|done|error.
    latest_only hides older versions of regenerated recordings."""
    return await _call(lambda: [_job_brief(j) for j in _client().list_jobs(query, status, latest_only, limit)])


@mcp.tool(annotations=READ)
async def get_transcript(job_id: str, kind: Literal["best", "raw", "clean"] = "best",
                         with_segments: bool = False, offset: int = 0, max_chars: int = 50000) -> str:
    """A transcript's text and metadata. best = AI-cleaned if available, else raw.
    with_segments adds timestamps. Long texts are paged with offset/max_chars."""
    def run() -> Dict[str, Any]:
        c = _client()
        job = c.get_job(job_id, text=True)
        raw, clean = job.pop("transcript_raw", None), job.pop("transcript_clean", None)
        text = {"raw": raw, "clean": clean}.get(kind) if kind != "best" else (clean or raw)
        out: Dict[str, Any] = {"job": _job_brief(job), "description": job.get("description"),
                               "settings": job.get("settings"), "kind": kind if kind != "best" else
                               ("clean" if clean else "raw")}
        if text is None:
            out["text"] = None
            out["note"] = (f"No {kind} text (status {job['status']}"
                           + (", AI finishing not run" if kind == "clean" else "") + ").")
        else:
            out["text"] = text[offset:offset + max_chars]
            out["total_chars"] = len(text)
            if offset + max_chars < len(text):
                out["next_offset"] = offset + max_chars
        if with_segments and job["status"] == "done":
            out["segments"] = c.segments(job_id)
        return out
    return await _call(run)


@mcp.tool(annotations=READ)
async def list_versions(job_id: str) -> str:
    """Every version of a recording (original + regenerations) with the settings each used."""
    return await _call(lambda: _client().versions(job_id))


@mcp.tool(annotations=WRITE)
async def regenerate_transcripts(job_ids: List[str],
                                 preset: Optional[Literal["accuracy", "balanced", "fast"]] = None,
                                 options: Optional[Dict[str, Any]] = None, keep_base: bool = True) -> str:
    """Re-transcribe recordings with other settings. Each run is a new version; the old one is kept.
    keep_base=true starts from the job's own settings and changes only what you pass."""
    return await _call(lambda: _client().action("regenerate", job_ids, preset=preset,
                                                options=options or {}, keep_base=keep_base))


@mcp.tool(annotations=WRITE)
async def run_ai_finishing(job_ids: List[str]) -> str:
    """Clean the text (fillers, false starts, Whisper artefacts) and write title, description and tags.
    Needs the OpenAI connection set in the web app. The raw transcript is never changed."""
    return await _call(lambda: _client().action("ai", job_ids))


@mcp.tool(annotations=WRITE)
async def send_to_supabase(job_ids: List[str]) -> str:
    """Queue transcripts for upsert into the configured Supabase table (retries until delivered)."""
    return await _call(lambda: _client().action("sync", job_ids))


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
