# Audio/Video to Transcript

A small Python tool that extracts audio from a video file and sends it to an OpenAI-compatible Transcriptions API (OpenAI, Groq, or the bundled local GPU server) to get a transcript, saving a `.txt` next to the original video. Runs as a CLI, a desktop GUI with a file queue, or a [browser app on a VPS](docs/deploy.md).

## Features
- Extract audio from most video formats via ffmpeg
- Silence-aware chunking: audio is split into short (~25–40s) segments at natural pauses, transcribed in parallel, and rejoined. Short chunks let Whisper re-detect the language on each utterance, which keeps multilingual meetings (e.g. Romanian + Ukrainian + Russian) from being garbled by a single wrong language
- Confidence filtering: no-speech and repetition-loop hallucinations are dropped using Whisper's per-segment stats (Groq `whisper-large-v3`), and every drop is logged
- Calls an OpenAI-compatible Transcriptions API — Groq (default model: `whisper-large-v3`) or OpenAI (default model: `gpt-4o-transcribe`)
- Or runs fully local: `gpu_server/` serves Whisper `large-v3` on your own NVIDIA GPU (faster-whisper) with the same API, no per-minute cost — usable from the web app and, over Tailscale, from the desktop GUI on another machine
- Transcribes in the spoken language: Whisper auto-detects the language of the recording (Russian, Romanian, …) and the transcript stays in that language for best accuracy
- Saves transcript in the same folder as the input video
- Desktop GUI: select a finished file to copy its transcript to the clipboard, open the `.txt`, or reveal it in the file manager (no console window flashes when ffmpeg runs)

## Requirements
- Python 3.9+
- ffmpeg available on PATH (https://ffmpeg.org/download.html)
- A Groq or OpenAI API key with access to transcriptions, **or** a reachable local GPU server (see [docs/deploy.md](docs/deploy.md))

## Setup (Windows PowerShell)
```powershell
# Download the app and enter its folder
git clone https://github.com/tanurus/video-transcriber.git
cd video-transcriber

# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your API key (temporarily for this session)
$env:GROQ_API_KEY = "gsk_..."   # or $env:OPENAI_API_KEY = "sk-..."

# Optional: Or create a .env file (copied from .env.example) to load automatically
Copy-Item .env.example .env
# then edit .env to add your key

# 4) Ensure ffmpeg is installed and on PATH
# Option A: If you use winget (Windows 11+)
winget install --id=Gyan.FFmpeg  --accept-package-agreements --accept-source-agreements
# Option B: Using Chocolatey (if installed)
choco install ffmpeg -y
# Option C: Manual — download a static build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH

# 5) Run the CLI against your file
python -m app.main "C:\path\to\your video.mkv"

# Or launch the desktop GUI (add files to a queue, watch progress)
python -m app.main --gui
```

The transcript is written as a `.txt` file with the same base name in the same folder as the video.

The desktop GUI includes a language selector, a **Settings...** dialog for chunk
sizes, silence sensitivity, and parallel requests, and **Remove selected** for
queued files. Settings apply to the next run and are not saved across restarts;
use `.env` for persistent defaults. **Auto (RO/RU/EN)** compares three candidate
languages on Whisper models and uses about three API calls per chunk. Other
models use a single transcription request without candidate comparison.

To update an existing installation, run `git pull --ff-only` in the app folder,
activate its virtual environment, and run `pip install -r requirements.txt`.
On a new machine, create `.env` from `.env.example` and enter your API key;
your local `.env` is not stored in GitHub.

## Configuration
You can customize behavior via environment variables (in your shell or `.env`, see `.env.example`):

- `GROQ_API_KEY` / `OPENAI_API_KEY`: If both are set, `GROQ_API_KEY` takes precedence and requests go to Groq's API. Not needed when using a local GPU server.
- `LOCAL_WHISPER_URL` (optional): URL of the bundled GPU server, e.g. `http://127.0.0.1:18921/v1` or `http://<machine>.<tailnet>.ts.net:18921/v1`. With it set, the local server is used whenever its `/health` answers; otherwise a cloud key is used if present.
- `TRANSCRIBE_PROVIDER` (optional): `auto` (default) | `local` | `groq` | `openai` — force one backend.
- `LOCAL_WHISPER_API_KEY` (optional): only if the GPU server was started with `WHISPER_API_KEY`.
- `TRANSCRIBE_MODEL` (optional): Defaults to `whisper-large-v3` on Groq, `gpt-4o-transcribe` on OpenAI — the highest-accuracy option on each provider. Make sure the model you set exists on the provider in use.
- `TRANSCRIBE_LANGUAGE` (optional): ISO-639-1 code (e.g. `ru`, `ro`) used as a hint when you know the spoken language up front. Leave unset to let Whisper auto-detect.
- `CANDIDATE_LANGUAGES` (optional): Comma-separated language codes, such as `ro,ru,en`. On Whisper models, transcribe each chunk once per candidate and select the highest-confidence result. This takes precedence over `TRANSCRIBE_LANGUAGE` and increases API calls and processing time. Leave unset to disable; non-Whisper models use the normal single request.
- `OPENAI_TIMEOUT` (optional): Request timeout in seconds, default 600.
- `AUDIO_BITRATE` (optional): MP3 bitrate used for export (e.g., `64k`, `96k`, `128k`). Default `96k`.
- `CHUNK_TARGET_MB` (optional): Safety-net max chunk size in MB (default 24). Only used if silence-aware chunking fails.

Silence-aware chunking (leave `TRANSCRIBE_LANGUAGE` unset for mixed-language audio):
- `CHUNK_TARGET_SEC` (default 25): aim to cut a segment around here, snapped to the nearest pause.
- `CHUNK_MAX_SEC` (default 40): force a cut by here even without a pause.
- `SILENCE_NOISE_DB` (default `-30dB`) / `SILENCE_MIN_SEC` (default 0.5): `silencedetect` tuning; raise the dB (e.g. `-25dB`) for noisier rooms.
- `MAX_CONCURRENCY` (default 4): how many segments to transcribe in parallel (keep modest for Groq rate limits).

Hallucination filtering (whisper models only): `NO_SPEECH_THRESHOLD` (0.6), `LOGPROB_THRESHOLD` (-1.0), `COMPRESSION_RATIO_THRESHOLD` (2.4) — Whisper's own defaults, override to tune.

Note: the transcript is written in the language spoken in the recording. Transcribing natively is more accurate than translating to English on the fly. For a mixed-language recording, keep everything native and let a downstream LLM translate/summarize the finished transcript — that preserves far more than Whisper's translate mode.

## Web app on a VPS (browser access over Tailscale)

A browser app you can use from any device on your tailnet — phone included.
If the VPS has an NVIDIA GPU, the bundled `gpu_server/` transcribes with Whisper
large-v3 locally instead of a paid API. See [docs/deploy.md](docs/deploy.md).

- **Upload** any number of files at once (drag-and-drop or the phone's file picker).
  Big files go up in resumable 8 MB pieces that retry on a flaky connection.
- **Settings that matter**, with presets (Maximum accuracy / Balanced / Fast) and an
  advanced panel: audio preparation (lossless audio, EBU R128 loudness, optional
  denoise), segmentation (smart chunks or Whisper-native VAD), Whisper decoding
  (beam, best-of, patience, temperature fallback, repetition controls, hotwords,
  prompt) and hallucination filters. **Profiles** store prompt, hotwords and
  languages for recurring meetings.
- **Library**: search, filter, select all, then send to Supabase, run AI finishing,
  regenerate with other settings, download a zip, or delete. Every regeneration is
  a new version; the lossless audio is kept so it works after the video is purged.
- **AI finishing** (any OpenAI-compatible endpoint): cleans fillers, false starts
  and Whisper hallucinations without translating, then writes a title, description
  and tags and renames the transcript. A guard keeps the original wording of any
  chunk the model rewrote instead of cleaning. The raw transcript is never changed.
- **Supabase**: every transcript (raw + clean text, timestamped segments, languages,
  device, recording time, settings, SHA-256, versions) is upserted into one table,
  with retries and backoff until it lands. Settings show the setup SQL.
- **API for other sources** (`POST /api/v1/jobs` with a token): phone shortcuts,
  n8n, scripts. The same recording sent from two devices is recognised by its
  SHA-256 and not transcribed twice.

### Using a VPS GPU from the desktop GUI

Point the desktop app at the VPS GPU server over Tailscale by adding to `.env`:

```
LOCAL_WHISPER_URL=http://<machine>.<tailnet>.ts.net:18921/v1
```

With a `GROQ_API_KEY` also set, the GUI falls back to Groq whenever the VPS is
unreachable. Each run's log names the backend and model that were used.

## Notes
- ffmpeg is required for extracting audio and chunking. If not found, install it with winget/choco or manually and reopen your terminal so PATH updates.
- The app uses temporary directories for intermediate audio/chunks. They are cleaned up automatically.
- Be mindful of API usage costs.

## Troubleshooting
- PowerShell quoting: if a command errors with "string is missing the terminator", retype the quotes and ensure they are straight quotes `"` and not curly ones. You can also avoid quotes by renaming the file to not include spaces, or by using the short path syntax.
- ffmpeg not found: after installing ffmpeg, close and reopen your terminal window so PATH changes take effect.
