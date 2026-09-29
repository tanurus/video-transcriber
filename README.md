# Audio/Video to Transcript

A small Python tool that extracts audio from a video file and sends it to an OpenAI-compatible Transcriptions API (OpenAI or Groq) to get a transcript, saving a `.txt` next to the original video. Runs as a CLI, a desktop GUI with a file queue, or a [browser app on a VPS](docs/deploy.md).

## Features
- Extract audio from most video formats via ffmpeg
- Silence-aware chunking: audio is split into short (~25–40s) segments at natural pauses, transcribed in parallel, and rejoined. Short chunks let Whisper re-detect the language on each utterance, which keeps multilingual meetings (e.g. Romanian + Ukrainian + Russian) from being garbled by a single wrong language
- Confidence filtering: no-speech and repetition-loop hallucinations are dropped using Whisper's per-segment stats (Groq `whisper-large-v3`), and every drop is logged
- Calls an OpenAI-compatible Transcriptions API — Groq (default model: `whisper-large-v3`) or OpenAI (default model: `gpt-4o-transcribe`)
- Transcribes in the spoken language: Whisper auto-detects the language of the recording (Russian, Romanian, …) and the transcript stays in that language for best accuracy
- Saves transcript in the same folder as the input video
- Desktop GUI: select a finished file to copy its transcript to the clipboard, open the `.txt`, or reveal it in the file manager (no console window flashes when ffmpeg runs)

## Requirements
- Python 3.9+
- ffmpeg available on PATH (https://ffmpeg.org/download.html)
- A Groq or OpenAI API key with access to transcriptions

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

- `GROQ_API_KEY` / `OPENAI_API_KEY` (one required): If both are set, `GROQ_API_KEY` takes precedence and requests go to Groq's API.
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

You can run this as a browser app on a Linux VPS and reach it from any device on
your Tailscale tailnet — upload a video, watch live progress, and download the
transcript, with no local app running.

See [docs/deploy.md](docs/deploy.md) for full instructions. In short:

```bash
git clone https://github.com/tanurus/video-transcriber.git && cd video-transcriber
echo 'GROQ_API_KEY=gsk_...' > .env
chmod +x deploy.sh && ./deploy.sh
sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
```

Then open `https://transcribe.<your-tailnet>.ts.net`.

## Notes
- ffmpeg is required for extracting audio and chunking. If not found, install it with winget/choco or manually and reopen your terminal so PATH updates.
- The app uses temporary directories for intermediate audio/chunks. They are cleaned up automatically.
- Be mindful of API usage costs.

## Troubleshooting
- PowerShell quoting: if a command errors with "string is missing the terminator", retype the quotes and ensure they are straight quotes `"` and not curly ones. You can also avoid quotes by renaming the file to not include spaces, or by using the short path syntax.
- ffmpeg not found: after installing ffmpeg, close and reopen your terminal window so PATH changes take effect.
