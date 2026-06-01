# VPS Web Transcriber — Design

**Date:** 2026-06-01
**Status:** Approved (pending written-spec review)

## Goal

Run the existing audio/video transcription tool as a web app on a Linux VPS, reachable
from any of the user's devices through their private Tailscale tailnet at a short,
explicit domain. From the browser the user uploads one video, watches live progress,
and downloads the resulting transcript as a `.txt`. Transcripts are kept on the VPS
permanently; uploaded videos are kept but auto-purged once they pass a configurable
age. No machine-local app needs to be running.

## Non-goals (v1)

- No public internet exposure (Tailscale Funnel is out; tailnet-only).
- No multi-file queue in the web UI (single file at a time; the desktop app keeps its queue).
- No app-level login/password (the private tailnet is the access boundary).
- No multi-user accounts, sharing, or permissions.
- No change to the existing Tkinter GUI or CLI behavior.

## Constraints / context

- Transcription runs through the **Groq** Whisper API (cloud), so the VPS needs no GPU —
  only Python + ffmpeg. `GROQ_API_KEY` lives in `.env` on the VPS.
- The VPS already has Docker available and Tailscale installed and joined to the tailnet.
- **`.env` must never be committed to git or baked into the Docker image.** It is mounted
  at runtime via compose `env_file`. (`.gitignore` already excludes `.env`.)
- Deployment is performed by the **user running a deploy script** over SSH; the assistant
  does not SSH into the VPS.

## Reuse of existing code

The core pipeline is already cleanly separated and is reused unchanged in spirit:

- `app/main.py::transcribe_video(video_path, cfg, logger, ...)` — extract → chunk → Groq → save.
- `app/audio.py` — `extract_audio`, `chunk_audio_by_size` (ffmpeg).
- `app/whisper_client.py::WhisperClient` — Groq/OpenAI-compatible client.
- `app/config.py::Config.load()` — reads `GROQ_API_KEY` / `OPENAI_API_KEY` from env/`.env`.

**One backward-compatible enhancement:** add an optional `out_path: Path | None = None`
parameter to `transcribe_video()`. When provided, the transcript is written there;
when omitted, behavior is unchanged (written next to the video). This lets the web worker
place transcripts directly in `/data/transcripts/<job_id>.txt` without touching the CLI/GUI.

## Architecture & topology

```
Your devices ──(Tailscale tailnet, HTTPS)──► VPS host
                                              │
                          tailscale serve :443 ─► 127.0.0.1:8000
                                              │
                                   ┌──────────▼────────────┐
                                   │ Docker service "web"  │
                                   │  gunicorn → Flask      │
                                   │  ffmpeg (in image)     │
                                   │  1 worker, N threads   │
                                   │  + 1 background worker  │
                                   │    (1 transcription at  │
                                   │     a time)             │
                                   └──────────┬────────────┘
                                              │  bind mount
                                        /data (host) + .env (host)
```

- **One Docker container** (`docker-compose.yml`, service `web`): gunicorn serving the Flask
  app, ffmpeg baked into the image.
- Container port published to **`127.0.0.1:8000` only** (never `0.0.0.0`) so it is not exposed
  on the public internet. All access is via Tailscale.
- **`tailscale serve`** on the host proxies HTTPS (:443) to `127.0.0.1:8000`. The VPS machine is
  renamed to `transcribe`, yielding `https://transcribe.<tailnet>.ts.net` (tailnet-only).
- **gunicorn runs a single worker with multiple threads** so the in-process job registry and the
  single background worker are coherent (no cross-process state). One transcription runs at a time
  via `ThreadPoolExecutor(max_workers=1)`; additional uploads queue.
- **gunicorn `--timeout` is set high (e.g. 1200s)** because the upload transfer happens inside the
  `POST /upload` request; the transcription itself runs in the background thread, so only the upload
  (not the transcription) is bounded by the request timeout. Threaded worker so a long upload does
  not block the polling endpoints.

## Storage layout

A host directory is bind-mounted into the container at `/data` and survives restarts/rebuilds:

```
/data/
  app.db                          # SQLite: job history
  uploads/<job_id>/<filename>     # uploaded video; auto-deleted after RETAIN_VIDEO_DAYS
  transcripts/<job_id>.txt        # transcript; kept forever
```

The host `.env` (with `GROQ_API_KEY`) is provided to the container via compose `env_file`.

### SQLite schema (`jobs` table)

| column         | type    | notes                                            |
|----------------|---------|--------------------------------------------------|
| id             | TEXT PK | job id (uuid hex)                                |
| original_name  | TEXT    | original uploaded filename                       |
| status         | TEXT    | queued \| running \| done \| error \| interrupted |
| created_at     | TEXT    | ISO timestamp (UTC)                              |
| completed_at   | TEXT    | ISO timestamp (UTC), nullable                    |
| transcript_path| TEXT    | relative path under /data, nullable              |
| error          | TEXT    | error message, nullable                          |

(SQLite is Python stdlib — no extra service. Single-writer access from the one worker process.)

## Web app (`web/` package)

Proposed module layout, each file with one clear responsibility:

- `web/__init__.py` — Flask app factory (`create_app`), config wiring, startup reconciliation.
- `web/storage.py` — SQLite access (create schema, insert/update/list/get job).
- `web/jobs.py` — job manager: executor, per-job in-memory log buffers, run function that
  calls `transcribe_video`, status transitions, retention/cleanup thread.
- `web/routes.py` — Flask blueprint with the HTTP endpoints.
- `web/templates/` — `index.html` (upload + recent list), `job.html` (live progress), `history.html`.
- `web/static/` — minimal CSS + a small `job.js` poller.
- `web/__main__` / `wsgi.py` — gunicorn entrypoint exposing the WSGI `app`.

### Routes

| Method | Path                  | Purpose                                                            |
|--------|-----------------------|-------------------------------------------------------------------|
| GET    | `/`                   | Upload page (file picker + Start) and a short recent-transcripts list. |
| POST   | `/upload`             | Validate type/size, save video, create job (`queued`), enqueue, redirect to `/job/<id>`. |
| GET    | `/job/<id>`           | Progress page; JS polls the API.                                  |
| GET    | `/api/job/<id>?since=N` | JSON: `{status, lines:[...], next_index, download_ready, error}`. |
| GET    | `/download/<id>`      | Serve `transcripts/<id>.txt` as an attachment.                    |
| GET    | `/history`            | All past jobs, newest first, with download links.                |
| GET    | `/healthz`            | Liveness probe (returns 200 + ok).                                |

### Job lifecycle & data flow

```
POST /upload
  → validate (extension in allowed set, size <= MAX_CONTENT_LENGTH)
  → save to /data/uploads/<id>/<filename>
  → SQLite insert (status=queued)
  → executor.submit(run_job, id)
  → redirect to /job/<id>

run_job(id):
  → status=running
  → transcribe_video(video_path, cfg, logger=buffer_appender(id),
                     out_path=/data/transcripts/<id>.txt)
  → on success: status=done, completed_at=now, transcript_path set
  → on exception: status=error, error=message
  (video is left in place; retention thread purges it later)

/job/<id> page polls /api/job/<id>?since=N every ~1s
  → renders streamed log lines; when download_ready, shows Download button + transcript preview.
```

### Live progress mechanism

The `logger` callback passed to `transcribe_video` appends each line to an in-memory,
lock-guarded list keyed by `job_id`. `/api/job/<id>?since=N` returns lines from index `N`
onward plus `next_index`. The page polls roughly once per second. Log buffers are ephemeral
(memory only); after a restart, completed jobs still show their final status from SQLite, but
the streamed log of an old job is not retained (acceptable for a personal tool).

## Retention / cleanup

A daemon thread started at app boot runs a cleanup pass at startup and every 24h:
delete any directory under `/data/uploads/` whose mtime is older than `RETAIN_VIDEO_DAYS`
(default 30). Transcripts under `/data/transcripts/` are never deleted. The value is
configurable via env var.

## Configuration (env vars)

| Var                | Default | Meaning                                              |
|--------------------|---------|------------------------------------------------------|
| `GROQ_API_KEY`     | —       | Required (or `OPENAI_API_KEY`). From host `.env`.    |
| `TRANSCRIBE_MODEL` | (auto)  | Existing override, unchanged.                        |
| `DATA_DIR`         | `/data` | Root for db/uploads/transcripts.                     |
| `RETAIN_VIDEO_DAYS`| `30`    | Age after which uploaded videos are purged.          |
| `MAX_CONTENT_MB`   | `2048`  | Max upload size (maps to Flask `MAX_CONTENT_LENGTH`).|
| `ALLOWED_EXT`      | mp4,mkv,mov,avi,webm,m4a,mp3,wav | Accepted upload extensions.     |

## Error handling

- **Invalid/missing/oversized file** → error rendered on the upload page; no job created.
- **ffmpeg or Groq failure** → job `status=error` with the message shown on the job page.
  The core already degrades gracefully (ffmpeg-missing fallback, Groq retry/backoff).
- **Container/app restart mid-job** → on startup, any job left in `queued`/`running` is
  reconciled to `interrupted` so it is not a ghost. Its video remains for manual re-upload.
- **No API key at startup** → app logs a clear fatal message and exits non-zero (compose will
  show it as unhealthy).

## Security

- Access boundary is the **private tailnet** via `tailscale serve` (no Funnel). This is the
  standard pattern for a single-user internal tool; the network is the auth.
- The container binds only to `127.0.0.1` on the host; nothing is published publicly.
- `.env` is never committed nor imaged; it is mounted at runtime.
- If Funnel (public) is ever enabled later, add an app password / basic auth first — explicitly
  out of scope for v1.

## Testing

- **Unit (no network/ffmpeg):**
  - `storage.py`: create schema, insert/update/list/get round-trips.
  - `jobs.py`: job lifecycle using a **fake transcriber** injected in place of `transcribe_video`
    (asserts status transitions, log buffering, error capture) — no real Groq/ffmpeg.
  - upload validation (extension + size rejection).
- **Integration (opt-in, marked):** run one tiny real audio clip through the worker, assert a
  `.txt` appears in `/data/transcripts` and the job reaches `done`.
- **Smoke:** `GET /healthz` returns 200; `GET /` renders.

## Deliverables (files added to the repo)

- `web/` package (modules + templates + static, as above).
- `app/main.py` — add optional `out_path` to `transcribe_video` (backward compatible).
- `Dockerfile` — Python base, `apt-get install ffmpeg`, install deps, run gunicorn.
- `docker-compose.yml` — service `web`, `env_file: .env`, bind mounts `./data:/data`, publish
  `127.0.0.1:8000:8000`, restart policy.
- `.dockerignore` — exclude `.venv`, `.git`, `data/`, `__pycache__`, `.env`, transcripts/videos.
- `requirements-web.txt` (or extend `requirements.txt`) — add `flask`, `gunicorn`.
- `deploy.sh` — idempotent VPS deploy script: pull, ensure `.env` present, build, `compose up -d`,
  print the `tailscale serve` command (and hostname rename hint).
- `docs/.../deploy.md` (or README section) — one-time setup + redeploy steps, Tailscale serve,
  machine rename to `transcribe`.
- Tests under `tests/`.

## Deployment flow

1. One-time on the VPS: clone the repo, create `/data`, create `.env` with `GROQ_API_KEY`,
   (optional) rename the Tailscale machine to `transcribe`.
2. `./deploy.sh` → `docker compose up -d --build`.
3. One-time: `tailscale serve --bg --https=443 http://127.0.0.1:8000`.
4. Browse to `https://transcribe.<tailnet>.ts.net`.
5. Redeploy after changes: `git pull && ./deploy.sh`.

## Open questions / future work

- Optional app password if Funnel is ever enabled.
- Optional "delete from history" button.
- Optional multi-file queue parity with the desktop app.
