# Deploying the Web Transcriber to a VPS

The web app runs as a Docker container reachable only over your Tailscale tailnet.
It can send audio to a cloud API (Groq/OpenAI) or to a **local GPU server**
(`gpu_server/`, faster-whisper) running on the same machine — or both, with the
local server preferred and the cloud as fallback.

```
Browser / desktop GUI (tailnet)
   │
   ├─► web container  :8000 (or your port)   upload → ffmpeg → silence-aware chunks
   │                                   │
   └───────────────────────────────────┴─► gpu_server  :18921   OpenAI-compatible
                                              /v1/audio/transcriptions on the GPU
```

## Prerequisites (on the VPS)
- Docker + the `docker compose` plugin
- Tailscale installed and logged in (`tailscale status` shows the machine)
- For the local GPU server: an NVIDIA GPU with a working driver (`nvidia-smi` works).
  ~5 GB of VRAM for `large-v3` in float16. No CUDA toolkit and no
  `nvidia-container-toolkit` needed — the server runs on the host, not in Docker,
  and gets the CUDA libraries from pip.

## 1. Local GPU server (optional, recommended if you have a GPU)

```bash
git clone https://github.com/tanurus/video-transcriber.git && cd video-transcriber
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -r gpu_server/requirements.txt
WHISPER_BIND="127.0.0.1:18921" gpu_server/run.sh      # first request downloads ~3 GB
curl http://127.0.0.1:18921/health                  # {"status": "ok", "cuda_devices": 1, ...}
```

Run it permanently as a user service (survives logout if lingering is on:
`loginctl enable-linger $USER`):

```ini
# ~/.config/systemd/user/whisper-gpu.service
[Service]
WorkingDirectory=%h/video-transcriber
Environment=WHISPER_VENV=%h/video-transcriber/.venv-gpu
Environment="WHISPER_BIND=127.0.0.1:18921 <tailnet-ip>:18921"
ExecStart=%h/video-transcriber/gpu_server/run.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now whisper-gpu
```

Binding the tailnet IP as well lets other tailnet machines (e.g. the Windows
desktop GUI) use this GPU by setting `LOCAL_WHISPER_URL=http://<machine>.<tailnet>.ts.net:18921/v1`.
Bind specific addresses, never `0.0.0.0`.

Server settings (environment):

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Default model; clients may also ask for any allowed one |
| `WHISPER_ALLOWED_MODELS` | `large-v3,large-v3-turbo` | Allowlist — other ids are rejected, never downloaded |
| `WHISPER_IDLE_UNLOAD_SEC` | `900` | Free VRAM after this long idle (0 = never) |
| `WHISPER_COMPUTE_TYPE` | `float16` | e.g. `int8_float16` for smaller GPUs |
| `WHISPER_BEAM_SIZE` | `5` | Beam search width |
| `WHISPER_API_KEY` | unset | If set, clients must send `Authorization: Bearer <key>` |
| `WHISPER_BIND` | `127.0.0.1:18921` | Space-separated `host:port` list |

Gotchas:
- `run.sh` puts the pip CUDA libraries (`site-packages/nvidia/*/lib`) on
  `LD_LIBRARY_PATH`; without that, ctranslate2 fails with
  `libcublas.so.12 is not found`.
- PyAV 19 breaks faster-whisper 1.2.1 (`metadata_errors` argument); the
  requirements pin `av<17`.
- One decode runs at a time; parallel requests queue. `MAX_CONCURRENCY=2` on the
  web side is plenty.

## 2. Web container

1. Create `.env` (never committed). Local GPU only:
   ```bash
   cat > .env <<'EOF'
   TRANSCRIBE_PROVIDER=local
   LOCAL_WHISPER_URL=http://127.0.0.1:18921/v1
   MAX_CONCURRENCY=2
   EOF
   ```
   Or cloud only: `GROQ_API_KEY=gsk_...`. Or both: set `LOCAL_WHISPER_URL` and a
   key, leave `TRANSCRIBE_PROVIDER` unset (`auto`) — local is used while its
   `/health` answers, the cloud otherwise.
2. The container must reach the GPU server. With the server on the host's
   loopback, run the container with `network_mode: host` (and drop `ports:`),
   binding gunicorn to specific addresses, e.g. in a compose override:
   ```yaml
   services:
     web:
       network_mode: host
       ports: !reset []
       command: ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "1200",
                 "--bind", "127.0.0.1:8000", "--bind", "<tailnet-ip>:8000", "web.wsgi:app"]
   ```
3. Deploy: `chmod +x deploy.sh && ./deploy.sh`
4. Open `http://<machine>.<tailnet>.ts.net:8000` from any tailnet device.

## Exposing it on the tailnet — read before using `tailscale serve`

- Binding the tailnet IP directly (as above) needs no root and no `tailscale serve`.
- If you want HTTPS via `tailscale serve`, use a port that is **not** already
  served, and check `tailscale funnel status` afterwards: Funnel is enabled per
  port, so adding this app to a port that already has Funnel on (often 443) makes
  it **public**. The app has no authentication by design.
- Do **not** rename the machine (`tailscale set --hostname=...`) on a host that
  runs other services — every existing `*.ts.net` URL for it changes.

## Redeploying after changes
```bash
./deploy.sh
```

## Where data lives
- `./data/app.db` — job history (SQLite)
- `./data/uploads/<job-id>/` — uploaded videos (auto-deleted after `RETAIN_VIDEO_DAYS`, default 30)
- `./data/audio/<job-id>.flac` — kept lossless 16 kHz audio per recording (never purged; makes regenerate work)
- `./data/segments/`, `./data/clean/`, `./data/ai/`, `./data/logs/` — timestamps, AI-cleaned text, AI metadata, per-job logs
- `./data/connections.json` (mode 0600) — OpenAI and Supabase settings entered in the UI; keys are never shown back
- `./data/incoming/` — unfinished chunked uploads (dropped after 48 h)
- `./data/transcripts/<job-id>.txt` — transcripts (kept forever)

## Configuration (set in `docker-compose.yml` or `.env`)
- `RETAIN_VIDEO_DAYS` (default 30)
- `MAX_CONTENT_MB` (default 2048)
- `ALLOWED_EXT` (default `mp4,mkv,mov,avi,webm,m4a,mp3,wav`)
- `SECRET_KEY` (optional) — only needed if you raise gunicorn workers above 1; otherwise a random per-process key is used for flash messages.
- Transcription variables (`TRANSCRIBE_PROVIDER`, `LOCAL_WHISPER_URL`, `TRANSCRIBE_MODEL`, `TRANSCRIBE_LANGUAGE`, `CHUNK_TARGET_SEC`, …) also apply — see the Configuration section in the [README](../README.md).

## Chunk length on a local GPU

On a cloud API every chunk is a paid request, so the defaults (25/40 s) keep the
count low. Locally a shorter chunk costs only GPU time, and it is the strongest
fix for mixed-language recordings: Whisper decodes a whole chunk in one
language. On a test clip switching Romanian / Russian / English every ~8 s,
25/40 s garbled about half the sentences while `CHUNK_TARGET_SEC=8`,
`CHUNK_MAX_SEC=15` got every sentence right (large-v3, ~12 s per 2 min of audio
on an RTX 4060 Ti).

## Connecting OpenAI and Supabase (once, in the UI)

Open **Settings → Connections**:

- **OpenAI** — API key, model (press *Test* to list the models your key can use)
  and the language for titles. Any OpenAI-compatible base URL works.
- **Supabase** — project URL, the `service_role` (or `sb_secret_…`) key and a table
  name. Run the **setup SQL** shown there once in the Supabase SQL editor, then press
  *Test*. The table has row-level security on with no policies, so only the service
  key can read or write it; add policies if you want to expose it.

Transcripts queue for Supabase automatically when "Send to Supabase" is on, or in
bulk from the library. Failed sends retry with backoff (1 min doubling to 1 h)
until they land; the library shows "Supabase: retrying" with the reason.

## Sending recordings from other devices

Any tailnet device can open the web app. For automatic sending, create a token
in **Settings → Other devices & API** and post files to `/api/v1/jobs`:

```bash
curl -H "Authorization: Bearer vt_..." -F file=@meeting.m4a -F preset=accuracy \
     -F source=phone http://<machine>.<tailnet>.ts.net:8000/api/v1/jobs
```

Optional fields: `profile_id`, `language`, `prompt`, `hotwords`, `title`,
`recorded_at` (ISO time), `options` (JSON of any setting), `on_duplicate=new`.
`GET /api/v1/jobs/<id>?text=1` returns status and text.

## Notes
- The web container never needs a public port; nothing is exposed to the internet.
- The web form offers **Best** (`whisper-large-v3`) or **Fast** (`whisper-large-v3-turbo`,
  quicker and slightly less accurate); Fast works on the local server and on Groq.
