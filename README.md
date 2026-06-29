# Audio/Video to Transcript

A small Python tool that extracts audio from a video file and sends it to an OpenAI-compatible Transcriptions API (OpenAI or Groq) to get a transcript, saving a `.txt` next to the original video. Runs as a CLI, a desktop GUI with a file queue, or a [browser app on a VPS](docs/deploy.md).

## Features
- Extract audio from most video formats via ffmpeg
- Automatically chunk large audio into smaller parts when needed
- Calls an OpenAI-compatible Transcriptions API — Groq (default model: `whisper-large-v3`) or OpenAI (default model: `gpt-4o-transcribe`)
- Output is always English: the translations endpoint transcribes and translates any spoken language (Russian, Romanian, …) into English
- Saves transcript in the same folder as the input video
- Desktop GUI: select a finished file to copy its transcript to the clipboard, open the `.txt`, or reveal it in the file manager

## Requirements
- Python 3.9+
- ffmpeg available on PATH (https://ffmpeg.org/download.html)
- A Groq or OpenAI API key with access to transcriptions

## Setup (Windows PowerShell)
```powershell
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

## Configuration
You can customize behavior via environment variables (in your shell or `.env`, see `.env.example`):

- `GROQ_API_KEY` / `OPENAI_API_KEY` (one required): If both are set, `GROQ_API_KEY` takes precedence and requests go to Groq's API.
- `TRANSCRIBE_MODEL` (optional): Defaults to `whisper-large-v3` on Groq, `gpt-4o-transcribe` on OpenAI. On Groq, keep `whisper-large-v3` — it is the only model that supports the translations endpoint. Make sure the model you set exists on the provider in use.
- `OPENAI_TIMEOUT` (optional): Request timeout in seconds, default 600.
- `CHUNK_TARGET_MB` (optional): Target max size per audio chunk before uploading, default 24 (MB).
- `AUDIO_BITRATE` (optional): MP3 bitrate used for export (e.g., `64k`, `96k`, `128k`). Default `96k`.

Note: output is always English. The app calls the translations endpoint, which translates any spoken language into English, so there is no language setting to configure.

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
